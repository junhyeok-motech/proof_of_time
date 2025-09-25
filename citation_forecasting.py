# emnlp_citation_forecasting_pilot.py
# Run: python emnlp_citation_forecasting_pilot.py
# Purpose: Pilot "Citation Forecasting" using EMNLP papers only.

import os, re, json, math, random, asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from jsonschema import Draft202012Validator, ValidationError
from litellm import acompletion

# -------------------- CONFIG --------------------
SEED = int(os.getenv("SEED", "2025"))
random.seed(SEED)

MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = 0
TIMEOUT = 120
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
MAX_TOKENS = 400
ENABLE_REPAIR_PASS = True

AWARDS_XLSX = os.getenv("AWARDS_XLSX", "data/pot-best-papers.xlsx")
HF_DATASET = "AIM-Harvard/EMNLP-Accepted-Papers"

# Pilot sampling targets (adjust freely)
RATIO_BEST = 0.08        # 5–10%
RATIO_OUTSTANDING = 0.12 # 10–15%
RATIO_MAIN = 0.43        # 40–45%
RATIO_FINDINGS = 0.37    # 35–40%
TOTAL_TARGET = int(os.getenv("TOTAL_TARGET", "400"))  # overall pilot sample size

# Years to include as "pre-2025" candidates
MAX_YEAR = int(os.getenv("MAX_YEAR", "2024"))

# Optional: enable simple verification against Semantic Scholar *now* (or run again in 2026)
ENABLE_S2_VERIFICATION = bool(int(os.getenv("ENABLE_S2_VERIFICATION", "0")))
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("SCHOLAR_API_KEY", None)

# -------------------- LLM PREDICTION --------------------
SYSTEM_PROMPT = """You are an expert bibliometrics analyst.
Given ONLY a paper title from EMNLP (pre-2025), predict whether its citations one year after publication will place it in the TOP-20% within its venue-year.

Return STRICT JSON matching the schema.
Rules:
- Decide based on signals in the title only: novelty, scope, foundationalness, method vs dataset vs system, and likely breadth of impact.
- Be conservative if uncertain.
- Provide a short rationale.

Schema:
{
  "will_be_top20_after_one_year": boolean,
  "confidence": number,  // 0.0-1.0
  "predicted_percentile": number, // 0-100 (lower is better rank; 5 means top-5%)
  "rationale": string
}
"""

JSON_SCHEMA = {
  "name": "citation_forecast",
  "schema": {
    "type": "object",
    "properties": {
      "will_be_top20_after_one_year": {"type": "boolean"},
      "confidence": {"type": "number"},
      "predicted_percentile": {"type": "number"},
      "rationale": {"type": "string"}
    },
    "required": ["will_be_top20_after_one_year", "confidence", "predicted_percentile"],
    "additionalProperties": False
  },
  "strict": True
}
JSON_VALIDATOR = Draft202012Validator(JSON_SCHEMA["schema"])

class TemporaryModelError(Exception):
    pass

def validate_or_raise(payload: Dict[str, Any]) -> None:
    JSON_VALIDATOR.validate(payload)

async def llm_once(title: str, repair_note: str = "") -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TITLE: {title.strip()}"}
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
        extra_body={"seed": 7}
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
async def call_model_with_repairs(title: str) -> Dict[str, Any]:
    try:
        return await llm_once(title)
    except TemporaryModelError as e:
        if ENABLE_REPAIR_PASS:
            return await llm_once(title, repair_note=str(e))
        raise

# -------------------- DATA LOADING & BUCKETING --------------------
@dataclass
class PaperRow:
    title: str
    year: Optional[int]
    track: Optional[str]
    area: Optional[str]
    acceptance_type: Optional[str]
    source_split: Optional[str]
    bucket: Optional[str]  # Best, Outstanding, Main, Findings

def normalize(s: Optional[str]) -> str:
    return (s or "").strip()

def safe_year(y) -> Optional[int]:
    try:
        return int(y)
    except Exception:
        return None

def load_emnlp_dataset() -> Dataset:
    # Loads all splits as a single concatenated dataset
    dsdict = load_dataset(HF_DATASET)
    ds_all = []
    for split, ds in dsdict.items():
        if "title" not in ds.column_names:
            continue
        ds = ds.filter(lambda ex: ex.get("title") is not None)
        ds = ds.add_column("source_split", [split]*len(ds))
        ds_all.append(ds)
    if not ds_all:
        raise RuntimeError("No EMNLP splits with 'title' found.")
    return concatenate_datasets(ds_all)

def load_awards_table(xlsx_path: str) -> pd.DataFrame:
    # Expect a table with at least columns for title and an award label (e.g., 'Best'/'Outstanding').
    df = pd.read_excel(xlsx_path)
    # Heuristic column detection
    cols = {c.lower(): c for c in df.columns}
    # Try to find title col
    title_col = cols.get("title") or next((c for c in df.columns if "title" in c.lower()), None)
    if title_col is None:
        raise RuntimeError("Awards XLSX must contain a title column.")
    df["__title__"] = df[title_col].astype(str).str.strip()

    # Award/Bucket column (best/outstanding/etc.)
    bucket_col = None
    for key in ["bucket", "award", "type", "category"]:
        if key in cols:
            bucket_col = cols[key]
            break
    if bucket_col is None:
        # default: try to infer from any 'best'/'outstanding' markers
        df["__bucket__"] = df["__title__"].apply(
            lambda t: "Best" if "best" in t.lower() else ("Outstanding" if "outstanding" in t.lower() else "Unknown")
        )
    else:
        df["__bucket__"] = df[bucket_col].astype(str).str.strip()
    return df[["__title__", "__bucket__"]].rename(columns={"__title__": "title", "__bucket__": "bucket"})

def title_key(t: str) -> str:
    # Lightweight normalization for matching titles across sources
    return re.sub(r"\W+", "", t.lower())

def split_into_buckets(
    ds_all: Dataset,
    awards_df: pd.DataFrame,
    total_target: int = TOTAL_TARGET
) -> Dict[str, Dataset]:
    # Filter candidate pool to pre-2025
    if "year" in ds_all.column_names:
        ds_all = ds_all.filter(lambda ex: (ex.get("year") is None) or (ex["year"] <= MAX_YEAR))
    # Prepare a lookup of award titles
    award_map = {title_key(r.title): r.bucket for r in awards_df.itertuples(index=False)}

    # Annotate bucket from awards where applicable
    def infer_bucket(ex):
        tk = title_key(ex["title"])
        bucket = award_map.get(tk)
        if bucket:
            # Normalize canonical labels
            if bucket.lower().startswith("best"):
                bucket = "Best"
            elif bucket.lower().startswith("out"):
                bucket = "Outstanding"
            else:
                bucket = bucket  # leave as-is
        return {"bucket": bucket or None}

    ds_all = ds_all.map(infer_bucket)
    # Split out known Best/Outstanding
    ds_best = ds_all.filter(lambda ex: ex.get("bucket") == "Best")
    ds_outs = ds_all.filter(lambda ex: ex.get("bucket") == "Outstanding")

    # Everything else = candidates for Main/Findings
    ds_remainder = ds_all.filter(lambda ex: ex.get("bucket") is None)

    # Try to detect findings vs main (if dataset has a 'track' or 'venue' flag indicating 'Findings')
    # Fallback: sample Findings randomly from remainder.
    def is_findings(ex):
        # Heuristic: track or source_split mentions 'findings'
        for key in ["track", "area", "acceptance_type", "source_split"]:
            v = (ex.get(key) or "").lower()
            if "finding" in v:
                return True
        # also if venue or collection field exists
        for key in ["venue", "collection", "booktitle"]:
            v = (ex.get(key) or "").lower()
            if "findings" in v:
                return True
        return False

    ds_findings_cand = ds_remainder.filter(is_findings)
    ds_main_cand = ds_remainder.filter(lambda ex: not is_findings(ex))

    # Compute target sizes
    n_total = min(total_target, len(ds_all))
    n_best = min(len(ds_best), math.ceil(n_total * RATIO_BEST))
    n_outs = min(len(ds_outs), math.ceil(n_total * RATIO_OUTSTANDING))
    n_find = min(len(ds_findings_cand), math.ceil(n_total * RATIO_FINDINGS))
    # main gets the rest (but cap by available)
    n_main = min(len(ds_main_cand), n_total - (n_best + n_outs + n_find))

    # Sample
    def sample_ds(ds, n):
        if n <= 0 or len(ds) == 0:
            return ds.select([])  # empty
        ids = list(range(len(ds)))
        random.shuffle(ids)
        ids = ids[:n]
        return ds.select(ids)

    ds_best_s = sample_ds(ds_best, n_best)
    ds_outs_s = sample_ds(ds_outs, n_outs)
    ds_find_s = sample_ds(ds_findings_cand, n_find)
    ds_main_s = sample_ds(ds_main_cand, n_main)

    # Label buckets explicitly
    def add_bucket_const(ds, name):
        if len(ds) == 0:
            return ds
        return ds.add_column("bucket", [name]*len(ds))

    ds_best_s = add_bucket_const(ds_best_s, "Best")
    ds_outs_s = add_bucket_const(ds_outs_s, "Outstanding")
    ds_find_s = add_bucket_const(ds_find_s, "Findings")
    ds_main_s = add_bucket_const(ds_main_s, "Main")

    return {
        "Best": ds_best_s,
        "Outstanding": ds_outs_s,
        "Findings": ds_find_s,
        "Main": ds_main_s
    }

# -------------------- OPTIONAL: Simple S2 Verification --------------------
if ENABLE_S2_VERIFICATION:
    import requests

def s2_get_paper_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort retrieval of a paper record by title from Semantic Scholar.
    WARNING: title-only search can be noisy.
    """
    if not ENABLE_S2_VERIFICATION:
        return None
    url = f"{S2_API}/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,year,citationCount,publicationDate,venue,authors,externalIds"
    }
    headers = {"accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    return data[0] if data else None

def crude_verify_now(pred: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    """
    Quick check (today) of citation count. In 2026, you should replace this with
    a proper venue-year percentile computation using a frozen snapshot.
    """
    if not ENABLE_S2_VERIFICATION:
        return None
    rec = s2_get_paper_by_title(title)
    if not rec:
        return {"ok": False, "reason": "not_found"}
    return {
        "ok": True,
        "title": rec.get("title"),
        "year": rec.get("year"),
        "citationCount_now": rec.get("citationCount"),
        "predicted_top20": pred.get("will_be_top20_after_one_year"),
        "predicted_percentile": pred.get("predicted_percentile"),
    }

# -------------------- PIPELINE (Predict) --------------------
async def predict_for_dataset(ds: Dataset) -> Dataset:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    will_top20 = [None]*len(ds)
    confidence = [None]*len(ds)
    pred_pct = [None]*len(ds)
    rationale = [None]*len(ds)

    async def worker(i, title):
        async with sem:
            res = await call_model_with_repairs(title)
            will_top20[i] = bool(res["will_be_top20_after_one_year"])
            confidence[i] = float(res.get("confidence", 0.0))
            pred_pct[i] = float(res.get("predicted_percentile", 50.0))
            rationale[i] = res.get("rationale", "")

    tasks = [worker(i, ds[i]["title"]) for i in range(len(ds))]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(ds), desc="LLM forecast"):
        await fut

    ds = ds.add_column("pred_will_top20", will_top20)
    ds = ds.add_column("pred_confidence", confidence)
    ds = ds.add_column("pred_percentile", pred_pct)
    ds = ds.add_column("pred_rationale", rationale)

    # Optional: quick verification now
    if ENABLE_S2_VERIFICATION:
        verif = []
        for i in range(len(ds)):
            v = crude_verify_now(
                {"will_be_top20_after_one_year": will_top20[i],
                 "predicted_percentile": pred_pct[i]},
                ds[i]["title"]
            )
            verif.append(v)
        ds = ds.add_column("s2_verify_now", verif)

    return ds

async def main():
    print("Loading EMNLP dataset…")
    ds_all = load_emnlp_dataset()

    print("Loading awards table…")
    awards_df = load_awards_table(AWARDS_XLSX)

    print("Splitting & sampling buckets…")
    buckets = split_into_buckets(ds_all, awards_df, total_target=TOTAL_TARGET)
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} rows")

    # Build a DatasetDict to keep buckets separated
    pilot = DatasetDict({k: v for k, v in buckets.items() if len(v) > 0})

    # Predict per bucket
    enriched = {}
    for name, ds in pilot.items():
        print(f"Predicting bucket: {name} (n={len(ds)})")
        enriched[name] = await predict_for_dataset(ds)

    out = DatasetDict(enriched)
    out_dir = "emnlp_citation_forecast_pilot"
    out.save_to_disk(out_dir)
    print(f"\nSaved enriched pilot dataset to: {out_dir}")

    # Also dump a JSONL per bucket for quick eyeballing
    os.makedirs("jsonl", exist_ok=True)
    for name, ds in out.items():
        path = os.path.join("jsonl", f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in ds.to_dict(batch_size=None):
                # to_dict returns columns->lists; reconstruct row-wise
                # Simpler: iterate over indices
                break
        # safer per-row write:
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(ds)):
                row = {k: ds[i][k] for k in ds.column_names}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {name} → {path}")

    # Print a tiny summary
    total = sum(len(ds) for ds in out.values())
    pos = sum(int(x) for ds in out.values() for x in ds["pred_will_top20"])
    print(f"\nTOTAL={total} | predicted top-20% positives={pos} ({pos/total:.1%})")

if __name__ == "__main__":
    asyncio.run(main())
