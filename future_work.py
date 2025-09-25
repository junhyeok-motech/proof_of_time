# future_work_with_verification.py

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datasets import Dataset
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jsonschema import Draft202012Validator, ValidationError
from litellm import acompletion

import requests  # for Semantic Scholar HTTP API

# -------------------- CONFIG --------------------
MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = 0
TIMEOUT = 120
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "6"))
MAX_TOKENS = 400
ENABLE_REPAIR_PASS = True

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.getenv("SCHOLAR_API_KEY", None)  # optional header

# -------------------- PROMPT & SCHEMA --------------------
SYSTEM_PROMPT = """You are an expert research analyst.
Given a professor’s name and affiliation, forecast their most likely research direction in 2025.

Output STRICT JSON matching the schema.

Rules:
- predicted_keywords: 3–10 concise future-oriented keywords.
- predicted_subfields: map to broad AI/ML subfields (e.g., "causal inference", "retrieval augmentation", "fairness").
- predicted_modalities: choose from {"text", "vision", "speech", "multimodal", "structured data", "policy/society"}.
- Provide a rationale field, 1–3 sentences grounded in their reputation or past expertise.
- Be conservative; do not speculate beyond known research identity.
"""

JSON_SCHEMA = {
    "name": "future_research_forecast",
    "schema": {
        "type": "object",
        "properties": {
            "professor": {"type": "string"},
            "affiliation": {"type": "string"},
            "predicted_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 10},
            "predicted_subfields": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "predicted_modalities": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        },
        "required": ["professor", "affiliation", "predicted_keywords", "predicted_subfields", "predicted_modalities"],
        "additionalProperties": False
    },
    "strict": True
}

JSON_VALIDATOR = Draft202012Validator(JSON_SCHEMA["schema"])

# -------------------- LLM Prediction Logic --------------------
class TemporaryModelError(Exception):
    pass

def validate_or_raise(payload: Dict[str, Any]) -> None:
    JSON_VALIDATOR.validate(payload)

async def llm_once(name: str, affiliation: str, repair_note: str = "") -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROFESSOR: {name}\nAFFILIATION: {affiliation}"}
    ]
    if repair_note:
        messages.append({
            "role": "system",
            "content": f"Previous response violated schema ({repair_note}). Return ONLY valid JSON."
        })

    resp = await acompletion(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        timeout=TIMEOUT,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA, "strict": True},
    )
    text = resp.choices[0].message["content"]
    if not text:
        raise TemporaryModelError("Empty LLM content")
    try:
        data = json.loads(text)
    except Exception:
        raise TemporaryModelError("Malformed JSON")
    try:
        validate_or_raise(data)
    except ValidationError as ve:
        raise TemporaryModelError(f"Schema validation failed: {ve.message}")
    return data

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type(TemporaryModelError)
)
async def call_model_with_repairs(name: str, affiliation: str) -> Dict[str, Any]:
    try:
        return await llm_once(name, affiliation)
    except TemporaryModelError as e:
        if ENABLE_REPAIR_PASS:
            return await llm_once(name, affiliation, repair_note=str(e))
        raise

# -------------------- Semantic Scholar Fetch & Verification --------------------
def get_author_id_by_name(name: str) -> Optional[str]:
    """
    Use Semantic Scholar search endpoint to find author ID by name.
    Returns first matching author_id or None.
    """
    url = f"{SEMANTIC_SCHOLAR_API}/author/search"
    params = {"query": name, "limit": 1}
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        return None
    data = resp.json()
    matches = data.get("data", [])
    if not matches:
        return None
    return matches[0].get("authorId")

def get_first_2025_paper(author_id: str) -> Optional[Dict[str, Any]]:
    """
    Given a Semantic Scholar authorId, fetch that author's papers,
    filter to publication year = 2025, and return the one with the earliest pub date (or any).
    Returns dict with title, abstract, year.
    """
    url = f"{SEMANTIC_SCHOLAR_API}/author/{author_id}/papers"
    params = {
        "fields": "title,abstract,year",
        "limit": 1000
    }
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        return None
    data = resp.json().get("data", [])
    # filter year == 2025
    papers_2025 = [p for p in data if p.get("year") == 2025]
    if not papers_2025:
        return None
    # choose one (e.g. first in list)
    return papers_2025[0]

def compare_prediction_with_paper(pred: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple alignment metrics:
    - keyword overlap: fraction of predicted_keywords appearing in title/abstract
    - subfield match: whether any predicted_subfields appear in title/abstract
    - modality check: whether modality (like "multimodal") is consistent with paper text (heuristic)
    - returns a report dict
    """
    title = paper.get("title", "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text = title + " " + abstract

    # keyword overlap count
    kw_pred = [kw.lower() for kw in pred.get("predicted_keywords", [])]
    matched_kw = [kw for kw in kw_pred if kw in text]
    frac_kw = len(matched_kw) / max(1, len(kw_pred))

    # subfield match (any)
    sf_pred = [sf.lower() for sf in pred.get("predicted_subfields", [])]
    matched_sf = [sf for sf in sf_pred if sf in text]
    subfield_hit = bool(matched_sf)

    # modality heuristic: check for modality words in text
    modalities = pred.get("predicted_modalities", [])
    modality_hits = []
    for m in modalities:
        if m.lower() in text:
            modality_hits.append(m)
    modality_match = bool(modality_hits)

    report = {
        "paper_title": paper.get("title"),
        "paper_year": paper.get("year"),
        "matched_keywords": matched_kw,
        "keyword_overlap_fraction": frac_kw,
        "matched_subfields": matched_sf,
        "subfield_match": subfield_hit,
        "modality_matched": modality_match,
        "matched_modalities": modality_hits,
    }
    return report

async def verify_prediction(name: str, affiliation: str, pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    For one professor: try to fetch their first 2025 paper and compare.
    Returns a dict with pred + verification report (or None if no paper).
    """
    # Step 1: find author_id
    author_id = get_author_id_by_name(name)
    if author_id is None:
        return {"prediction": pred, "verification": None, "error": "author_id not found"}

    # Step 2: fetch first 2025 paper
    paper = get_first_2025_paper(author_id)
    if paper is None:
        return {"prediction": pred, "verification": None, "error": "no 2025 paper found"}

    # Step 3: compare
    report = compare_prediction_with_paper(pred, paper)
    return {"prediction": pred, "verification": report}

# -------------------- Pipeline: Predict + Verify --------------------
async def process_and_verify(professors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    For each professor, produce prediction and attempt verification.
    """
    results = []
    # first get predictions in parallel
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(entry):
        async with sem:
            pred = await call_model_with_repairs(entry["professor"], entry["affiliation"])
            # synchronous verify
            verify_res = await asyncio.get_event_loop().run_in_executor(
                None, verify_prediction, entry["professor"], entry["affiliation"], pred
            )
            results.append(verify_res)

    tasks = [worker(entry) for entry in professors]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(professors), desc="Predict+Verify"):
        await fut

    return results

async def main():
    professors = [
        {"professor": "Regina Barzilay", "affiliation": "MIT"},
        {"professor": "Anton Korinek", "affiliation": "University of Virginia"},
        # ... add others
    ]

    results = await process_and_verify(professors)

    # Save to JSON for inspection
    with open("predictions_with_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Finished. Results saved to predictions_with_verification.json")

if __name__ == "__main__":
    asyncio.run(main())
