#!/usr/bin/env python3
# future_work_with_verification_logged.py
# LLM-as-a-Judge (forecast verification) — with logging, error handling, and timeout safety

import os
import json
import asyncio
import time
import aiohttp
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from asyncio import TimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jsonschema import Draft202012Validator
from litellm import acompletion
import requests
import logging
import sys
import traceback

# -------------------- LOGGING SETUP --------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging started: {LOG_FILE}")

# Optional: redirect print() → logger
class LoggerWriter:
    def __init__(self, level): self.level = level
    def write(self, message):
        message = message.strip()
        if message: self.level(message)
    def flush(self): pass
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

# -------------------- CONFIG --------------------
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = 0
TIMEOUT = 120
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
MAX_TOKENS = 400
FREQ_AWARE = False

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SCHOLAR_API_KEY", None)

# Async rate limiter
last_request_time = 0
rate_limit_lock = asyncio.Lock()
async def rate_limit(interval=1.0):
    global last_request_time
    async with rate_limit_lock:
        delta = time.time() - last_request_time
        if delta < interval:
            await asyncio.sleep(interval - delta)
        last_request_time = time.time()

# -------------------- TAXONOMY LOAD --------------------
with open("data/research_taxonomy.json", "r") as f:
    TAXONOMY = json.load(f)

FIELDS = TAXONOMY["fields"]
KEYWORDS = TAXONOMY["keywords"]
ALLOWED_EXTRA_TERMS = ["deep learning", "neural networks", "machine learning", "AI safety"]

# -------------------- PROMPT + SCHEMA --------------------
SYSTEM_PROMPT = f"""
You are an expert research analyst.
Given a professor’s name and affiliation (optional), forecast their most likely research direction in 2025.
Output STRICT JSON matching the schema.

Rules:
- predicted_keywords: 3–10 concise keywords strictly from the predefined keyword list.
- predicted_fields: 1–5 broad research areas strictly from the predefined field list.
- Be conservative; do not speculate beyond known research identity.
- Do NOT generate any terms outside the provided lists.

Allowed fields: {len(FIELDS)}
Allowed keywords: {len(KEYWORDS)}
"""

JSON_SCHEMA = {
    "name": "future_research_forecast",
    "schema": {
        "type": "object",
        "properties": {
            "professor": {"type": "string"},
            "affiliation": {"type": "string"},
            "predicted_keywords": {
                "type": "array",
                "items": {"type": "string", "enum": KEYWORDS + ALLOWED_EXTRA_TERMS},
                "minItems": 3, "maxItems": 10, "uniqueItems": True
            },
            "predicted_fields": {
                "type": "array",
                "items": {"type": "string", "enum": FIELDS},
                "minItems": 1, "maxItems": 5, "uniqueItems": True
            },
            "rationale": {"type": "string", "minLength": 20, "maxLength": 600}
        },
        "required": ["professor", "affiliation", "predicted_keywords",
                     "predicted_fields", "rationale"],
        "additionalProperties": False
    }
}
JSON_VALIDATOR = Draft202012Validator(JSON_SCHEMA["schema"])

# -------------------- HELPERS --------------------
def validate_or_raise(payload: Dict[str, Any]): JSON_VALIDATOR.validate(payload)

def sanitize_prediction(data):
    valid_kw = set(KEYWORDS + ALLOWED_EXTRA_TERMS)
    data["predicted_keywords"] = [k for k in data.get("predicted_keywords", []) if k in valid_kw]
    if len(data["predicted_keywords"]) < 3:
        data["predicted_keywords"] += ["representation learning"]
    return data

# -------------------- LLM CALL --------------------
async def llm_once(name: str, affiliation: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROFESSOR: {name}\nAFFILIATION: {affiliation}"}
    ]
    try:
        resp = await asyncio.wait_for(
            acompletion(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                timeout=TIMEOUT,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            ),
            timeout=150
        )
        data = json.loads(resp.choices[0].message["content"])
        data = sanitize_prediction(data)
        validate_or_raise(data)
        return data
    except Exception as e:
        logger.error(f"[LLM Error] {name}: {e}")
        raise

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=1, max=20))
async def call_model_with_repairs(name: str, affiliation: str) -> Dict[str, Any]:
    return await llm_once(name, affiliation)

# -------------------- SEMANTIC SCHOLAR --------------------
async def get_author_id_by_name(name: str, affiliation: str = "") -> Optional[str]:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {"query": name, "fields": "name,affiliations,url,citationCount"}
    await rate_limit()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SEMANTIC_SCHOLAR_API}/author/search", headers=headers, params=params, timeout=15) as resp:
                if resp.status != 200:
                    logger.warning(f"[HTTP {resp.status}] Author search failed for {name}")
                    return None
                data = await resp.json()
                results = data.get("data", [])
                if not results:
                    return None
                if affiliation:
                    for a in results:
                        if affiliation.lower() in str(a.get("affiliations", "")).lower():
                            return a["authorId"]
                sorted_results = sorted(results, key=lambda x: x.get("citationCount", 0), reverse=True)
                return sorted_results[0].get("authorId")
    except Exception as e:
        logger.error(f"[Error] get_author_id_by_name({name}): {e}")
        return None

async def get_all_2025_papers(author_id: str) -> List[Dict[str, Any]]:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    params = {"fields": "title,abstract,year,venue,url", "limit": 1000}
    await rate_limit()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers", headers=headers, params=params, timeout=20) as resp:
                if resp.status != 200:
                    logger.warning(f"[HTTP {resp.status}] Paper fetch failed for {author_id}")
                    return []
                data = await resp.json()
                return [p for p in data.get("data", []) if p.get("year") == 2025]
    except Exception as e:
        logger.error(f"[Error] get_all_2025_papers({author_id}): {e}")
        return []

# -------------------- METADATA EXTRACTION --------------------
async def extract_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    title = paper.get("title", "No title")
    abstract = paper.get("abstract", "No abstract available")

    extraction_prompt = f"""
You are an expert research analyst.
Given a research paper's title and abstract, extract structured metadata
based on predefined vocabularies.

PAPER:
Title: {title}
Abstract: {abstract}

Rules:
- actual_keywords: from keyword list
- actual_fields: from field list
- Be conservative; no new terms.

Allowed fields: {FIELDS}
Allowed keywords: {KEYWORDS}
"""

    schema = {
        "name": "paper_metadata",
        "schema": {
            "type": "object",
            "properties": {
                "actual_keywords": {"type": "array", "items": {"type": "string", "enum": KEYWORDS}, "minItems": 3},
                "actual_fields": {"type": "array", "items": {"type": "string", "enum": FIELDS}, "minItems": 1},
            },
            "required": ["actual_keywords", "actual_fields"],
        },
    }

    try:
        resp = await asyncio.wait_for(
            acompletion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0,
                max_tokens=400,
                response_format={"type": "json_schema", "json_schema": schema}
            ),
            timeout=90
        )
        data = json.loads(resp.choices[0].message["content"])
        data["title"] = title
        return data
    except Exception as e:
        logger.warning(f"[Metadata Error] '{title}': {e}")
        return {"actual_keywords": title.lower().split()[:5], "actual_fields": ["unknown"], "title": title}

# -------------------- EVALUATION --------------------
def evaluate_forecast_vs_actual(pred: Dict[str, Any], papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _score(pred: List[str], gold: List[str]):
        if not pred or not gold: return {"precision": 0, "recall": 0, "f1": 0}
        p, g = set(pred), set(gold)
        tp = len(p & g)
        prec, rec = tp / len(p), tp / len(g)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return {"precision": prec, "recall": rec, "f1": f1}

    all_kw, all_fields = set(), set()
    for p in papers:
        all_kw.update(p.get("actual_keywords", []))
        all_fields.update(p.get("actual_fields", []))

    agg_kw = _score(pred["predicted_keywords"], list(all_kw))
    agg_f = _score(pred["predicted_fields"], list(all_fields))
    composite = np.mean([agg_kw["f1"], agg_f["f1"]])
    verdict = "strong" if composite > 0.7 else "moderate" if composite > 0.4 else "weak"

    return {
        "aggregate": {
            "keywords": agg_kw,
            "fields": agg_f,
            "composite_f1": composite,
            "verdict": verdict,
            "actual_keywords": list(all_kw),
            "actual_fields": list(all_fields),
            "n_papers": len(papers)
        }
    }

# -------------------- VERIFY --------------------
async def verify_prediction(name: str, aff: str, pred: Dict[str, Any]) -> Dict[str, Any]:
    try:
        author_id = await asyncio.wait_for(get_author_id_by_name(name, aff), timeout=30)
        if not author_id:
            return {"prediction": pred, "verification": None, "error": "author not found"}
        papers = await asyncio.wait_for(get_all_2025_papers(author_id), timeout=30)
        if not papers:
            return {"prediction": pred, "verification": None, "error": "no papers"}
        metas = [await asyncio.wait_for(extract_paper_metadata(p), timeout=60) for p in papers]
        return {"prediction": pred, "verification": evaluate_forecast_vs_actual(pred, metas)}
    except Exception as e:
        logger.error(f"[Verify Error] {name}: {e}")
        return {"prediction": pred, "verification": None, "error": str(e)}

# -------------------- SAFE WRAPPERS --------------------
async def safe_call(coro, name: str, timeout: int = 120):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"[Timeout] {name} ({timeout}s)")
        return {"error": f"timeout after {timeout}s", "name": name}
    except Exception as e:
        logger.error(f"[Error] {name}: {e}\n{traceback.format_exc(limit=1)}")
        return {"error": str(e), "name": name}

# -------------------- PROCESS BATCH --------------------
async def process_and_verify(professors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async def worker(entry):
        async with sem:
            name, aff = entry["professor"], entry["affiliation"]
            logger.info(f"[Start] {name} ({aff})")
            pred = await safe_call(call_model_with_repairs(name, aff), f"forecast({name})", timeout=180)
            ver = await safe_call(verify_prediction(name, aff, pred), f"verify({name})", timeout=300)
            logger.info(f"[Done] {name}")
            return {"professor": name, "affiliation": aff, "result": ver}
    tasks = [worker(e) for e in professors]
    return await asyncio.gather(*tasks, return_exceptions=True)

# -------------------- LOAD PROFESSORS --------------------
def load_professors_from_json(file_path="data/influential_ai_ppl_list.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profs = []
    if "TIME100_AI_Professors" in data:
        for year_data in data["TIME100_AI_Professors"].values():
            for person in year_data.get("people", []):
                profs.append({"professor": person["name"], "affiliation": person["affiliation"]})
    if "AI2050_Fellows" in data:
        for year_data in data["AI2050_Fellows"].values():
            for year, level_data in year_data.items():
                for person in level_data:
                    profs.append({"professor": person, "affiliation": ""})
    return profs

# -------------------- MAIN --------------------
async def main():
    professors = load_professors_from_json()
    logger.info(f"Loaded {len(professors)} professors from file")
    results = await process_and_verify(professors)
    with open("future_work_results/predictions_with_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Finished. Results saved to predictions_with_verification.json")

if __name__ == "__main__":
    asyncio.run(main())
