# pip install datasets pandas tqdm requests python-dateutil aiohttp

import re
import json
import time
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import requests
from dateutil.parser import parse as dtparse
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
from datetime import datetime

NCT_RE = re.compile(r"^NCT\d{8}$")

# Global tracking dictionary
tracking = {
    "start_time": time.time(),
    "stages": {},
    "filtering_stats": {},
    "retrieval_stats": {},
    "current_stage": None
}

def start_stage(stage_name, description=""):
    """Start tracking a processing stage"""
    tracking["current_stage"] = stage_name
    tracking["stages"][stage_name] = {
        "description": description,
        "start_time": time.time(),
        "end_time": None,
        "duration": None
    }
    print(f"\n🚀 STAGE: {stage_name}")
    if description:
        print(f"   {description}")

def end_stage(stage_name=None):
    """End tracking a processing stage"""
    if stage_name is None:
        stage_name = tracking["current_stage"]
    
    if stage_name and stage_name in tracking["stages"]:
        end_time = time.time()
        tracking["stages"][stage_name]["end_time"] = end_time
        tracking["stages"][stage_name]["duration"] = end_time - tracking["stages"][stage_name]["start_time"]
        duration = tracking["stages"][stage_name]["duration"]
        print(f"✅ Completed {stage_name} in {duration:.2f} seconds")

def log_filtering(step_name, before_count, after_count, description=""):
    """Log filtering statistics"""
    filtered_count = before_count - after_count
    retention_rate = (after_count / before_count * 100) if before_count > 0 else 0
    
    tracking["filtering_stats"][step_name] = {
        "before": before_count,
        "after": after_count,
        "filtered_out": filtered_count,
        "retention_rate": retention_rate,
        "description": description
    }
    
    print(f"🔍 {step_name}:")
    print(f"   Before: {before_count:,} | After: {after_count:,} | Filtered: {filtered_count:,} ({100-retention_rate:.1f}%)")
    if description:
        print(f"   {description}")

def log_retrieval_progress(retrieved, total, start_time, stage="retrieval"):
    """Log retrieval progress with speed metrics"""
    elapsed = time.time() - start_time
    rate = retrieved / elapsed if elapsed > 0 else 0
    eta = (total - retrieved) / rate if rate > 0 else float('inf')
    
    tracking["retrieval_stats"][stage] = {
        "retrieved": retrieved,
        "total": total,
        "elapsed": elapsed,
        "rate_per_second": rate,
        "eta_seconds": eta if eta != float('inf') else None,
        "progress_percent": (retrieved / total * 100) if total > 0 else 0
    }

def print_current_status():
    """Print current processing status"""
    current_stage = tracking["current_stage"]
    total_elapsed = time.time() - tracking["start_time"]
    
    print(f"\n📊 CURRENT STATUS (Total elapsed: {total_elapsed:.1f}s)")
    print(f"   Current stage: {current_stage}")
    
    # Show filtering stats
    if tracking["filtering_stats"]:
        print("   Recent filtering:")
        for step, stats in list(tracking["filtering_stats"].items())[-3:]:  # Last 3 steps
            print(f"     {step}: {stats['after']:,} remaining ({stats['retention_rate']:.1f}% retained)")
    
    # Show retrieval stats
    if tracking["retrieval_stats"]:
        latest_retrieval = list(tracking["retrieval_stats"].values())[-1]
        if latest_retrieval.get("rate_per_second"):
            print(f"   Retrieval speed: {latest_retrieval['rate_per_second']:.1f} req/sec")

# ---------- 1) Load subsets ----------
start_stage("data_loading", "Loading TrialPanorama datasets from HuggingFace")

print("Loading datasets...")
# studies is sharded into two files
studies = load_dataset(
    "parquet",
    data_files=[
        "hf://datasets/zifeng-ai/TrialPanorama-database/studies.parquet",
        "hf://datasets/zifeng-ai/TrialPanorama-database/studies_part_2.parquet",
    ],
    split="train",   # gives you a single Dataset
)

# relations is a single file
relations = load_dataset(
    "parquet",
    data_files="hf://datasets/zifeng-ai/TrialPanorama-database/relations.parquet",
    split="train",
)

# Convert to DataFrame with progress tracking and memory optimization
print("Converting datasets to DataFrames...")
print(f"Studies dataset size: {len(studies):,} rows")
print(f"Relations dataset size: {len(relations):,} rows")

# Convert in chunks to avoid memory issues and show progress
print("Converting studies dataset...")
start_time = time.time()
df_studies = studies.to_pandas()  # More efficient than pd.DataFrame(studies)
studies_time = time.time() - start_time
print(f"Studies conversion completed in {studies_time:.2f} seconds")

print("Converting relations dataset...")
start_time = time.time()
df_rel = relations.to_pandas()  # More efficient than pd.DataFrame(relations)
relations_time = time.time() - start_time
print(f"Relations conversion completed in {relations_time:.2f} seconds")

print(f"Loaded {len(df_studies):,} studies and {len(df_rel):,} relations")
print("Studies columns:", df_studies.columns.tolist()[:10], "..." if len(df_studies.columns) > 10 else "")
print("Relations columns:", df_rel.columns.tolist()[:10], "..." if len(df_rel.columns) > 10 else "")

# Memory usage info
studies_memory = df_studies.memory_usage(deep=True).sum() / 1024**2  # MB
relations_memory = df_rel.memory_usage(deep=True).sum() / 1024**2   # MB
print(f"Memory usage - Studies: {studies_memory:.1f} MB, Relations: {relations_memory:.1f} MB")

end_stage()

# ---------- 2) Normalize column names ----------
start_stage("column_normalization", "Finding and normalizing key column names")

def get_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns {candidates} found in {df.columns.tolist()}")

# Find key columns
study_id_col = get_first_existing(df_studies, ["study_id", "id", "studyId", "study_identifier", "studyid"])
title_col = next((c for c in ["title", "study_title", "official_title", "brief_title"] if c in df_studies.columns), None)

# Relations columns
src_id_col = next((c for c in ["src_id","source_id","head_id","node1_id","from_id"] if c in df_rel.columns), None)
dst_id_col = next((c for c in ["dst_id","target_id","tail_id","node2_id","to_id"] if c in df_rel.columns), None)
src_type_col = next((c for c in ["src_type","source_type","head_type","node1_type","from_type"] if c in df_rel.columns), None)
dst_type_col = next((c for c in ["dst_type","target_type","tail_type","node2_type","to_type"] if c in df_rel.columns), None)

print(f"Using study ID column: {study_id_col}")
print(f"Using relation columns: {src_id_col} -> {dst_id_col}")

end_stage()

# ---------- 3) Initial deduplication ----------
start_stage("deduplication", "Removing duplicate studies and relations")

studies_before = len(df_studies)
relations_before = len(df_rel)

df_studies = df_studies.drop_duplicates(subset=[study_id_col])
df_rel = df_rel.drop_duplicates()

log_filtering("study_deduplication", studies_before, len(df_studies), "Removed duplicate studies")
log_filtering("relation_deduplication", relations_before, len(df_rel), "Removed duplicate relations")

print_current_status()
end_stage()

# ---------- 4) Find study-to-NCT mappings ----------
start_stage("nct_mapping", "Finding connections between studies and NCT IDs")

def is_nct(x):
    return isinstance(x, str) and bool(NCT_RE.match(x))

# A) Direct NCTs from studies
direct_nct = df_studies[[study_id_col]].copy()
direct_nct["nct_id"] = direct_nct[study_id_col].where(direct_nct[study_id_col].map(is_nct))
direct_nct_before = len(direct_nct)
direct_nct = direct_nct.dropna(subset=["nct_id"]).drop_duplicates()

log_filtering("direct_nct_extraction", direct_nct_before, len(direct_nct), "Studies with direct NCT IDs in study_id field")

# B) From relations - look for connections to NCT IDs
edges = []
if src_id_col and dst_id_col:
    # Forward direction
    tmp = df_rel[[src_id_col, dst_id_col]].dropna().drop_duplicates()
    tmp.columns = ["a", "b"]
    edges.append(tmp)
    
    # Reverse direction
    tmp2 = df_rel[[dst_id_col, src_id_col]].dropna().drop_duplicates()
    tmp2.columns = ["a", "b"]
    edges.append(tmp2)

if edges:
    edges_df = pd.concat(edges, ignore_index=True).drop_duplicates()
    edges_before = len(edges_df)
else:
    edges_df = pd.DataFrame(columns=["a", "b"])
    edges_before = 0

# Keep only edges where one end is an NCT ID
edges_nct = edges_df[(edges_df["a"].map(is_nct)) | (edges_df["b"].map(is_nct))].copy()

if edges_before > 0:
    log_filtering("nct_edge_filtering", edges_before, len(edges_nct), "Edges connecting to NCT IDs")

# Create (study_id, nct_id) pairs
def create_study_nct_pair(row):
    a, b = row["a"], row["b"]
    if is_nct(a) and not is_nct(b):
        return pd.Series({"study_like_id": b, "nct_id": a})
    elif is_nct(b) and not is_nct(a):
        return pd.Series({"study_like_id": a, "nct_id": b})
    else:
        return pd.Series({"study_like_id": None, "nct_id": None})

if not edges_nct.empty:
    pairs_before = len(edges_nct)
    pairs = edges_nct.apply(create_study_nct_pair, axis=1).dropna().drop_duplicates()
    
    log_filtering("study_nct_pair_creation", pairs_before, len(pairs), "Valid study-NCT pairs from relations")
    
    # Only keep pairs where study_like_id exists in our studies
    indirect_before = len(pairs)
    indirect_map = df_studies[[study_id_col]].merge(
        pairs, left_on=study_id_col, right_on="study_like_id", how="inner"
    )[[study_id_col, "nct_id"]].drop_duplicates()
    
    log_filtering("indirect_mapping_validation", indirect_before, len(indirect_map), "Pairs matching existing studies")
else:
    indirect_map = pd.DataFrame(columns=[study_id_col, "nct_id"])

print_current_status()
end_stage()

# ---------- 5) Combine and enforce bijection (1:1 mapping) ----------
start_stage("bijection_enforcement", "Creating 1:1 study-to-NCT mappings")

# Combine direct and indirect mappings
all_links = pd.concat([
    direct_nct[[study_id_col, "nct_id"]],
    indirect_map[[study_id_col, "nct_id"]]
], ignore_index=True).drop_duplicates()

print(f"Found {len(all_links):,} total study-to-NCT links")

# Enforce bijection: each study maps to exactly one NCT, each NCT maps from exactly one study
study_nct_counts = all_links.groupby(study_id_col)["nct_id"].nunique()
nct_study_counts = all_links.groupby("nct_id")[study_id_col].nunique()

# Keep only 1:1 mappings
valid_studies = study_nct_counts[study_nct_counts == 1].index
valid_ncts = nct_study_counts[nct_study_counts == 1].index

bijective_links_before = len(all_links)
bijective_links = all_links[
    all_links[study_id_col].isin(valid_studies) & 
    all_links["nct_id"].isin(valid_ncts)
].drop_duplicates()

log_filtering("bijection_filtering", bijective_links_before, len(bijective_links), "Enforcing 1:1 study-NCT mapping")

# Add study metadata
if title_col:
    bijective_links = bijective_links.merge(
        df_studies[[study_id_col, title_col]].drop_duplicates(), 
        on=study_id_col, 
        how="left"
    )

print_current_status()
end_stage()

# ---------- 6) FAST ClinicalTrials.gov data retrieval ----------
API_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Method 1: Async approach (fastest for large datasets)
async def fetch_ctgov_async(session, nct_id, semaphore):
    """Async fetch with semaphore to limit concurrent requests"""
    async with semaphore:
        url = f"{API_BASE}/{nct_id}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return nct_id, data, None
                else:
                    return nct_id, None, f"HTTP {response.status}"
        except asyncio.TimeoutError:
            return nct_id, None, "Timeout"
        except Exception as e:
            return nct_id, None, str(e)

async def fetch_all_async(nct_ids, max_concurrent=50):
    """Fetch all NCT data using async requests with progress tracking"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection pool size
        limit_per_host=50,  # Max connections per host
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=timeout,
        headers={'User-Agent': 'TrialPanorama-Processor/1.0'}
    ) as session:
        
        tasks = [fetch_ctgov_async(session, nct_id, semaphore) for nct_id in nct_ids]
        
        results = []
        failed = []
        
        # Track progress with detailed metrics
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=len(tasks), desc="Fetching CT.gov (async)", unit="req")
        
        # Process results as they complete
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            nct_id, data, error = await coro
            if data:
                results.append((nct_id, data))
            else:
                failed.append((nct_id, error))
            
            # Update progress every 50 requests
            pbar.update(1)
            if (i + 1) % 50 == 0 or i == len(tasks) - 1:
                log_retrieval_progress(i + 1, len(tasks), start_time, "async_fetch")
                current_rate = tracking["retrieval_stats"]["async_fetch"]["rate_per_second"]
                pbar.set_postfix({"rate": f"{current_rate:.1f}/s", "success": len(results), "failed": len(failed)})
        
        pbar.close()
        return results, failed

# Method 2: ThreadPool approach (good fallback, works in more environments)
def fetch_ctgov_sync(session, nct_id):
    """Synchronous fetch using requests session"""
    url = f"{API_BASE}/{nct_id}"
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200:
            return nct_id, response.json(), None
        else:
            return nct_id, None, f"HTTP {response.status_code}"
    except Exception as e:
        return nct_id, None, str(e)

def fetch_all_threaded(nct_ids, max_workers=20):
    """Fetch all NCT data using ThreadPoolExecutor with progress tracking"""
    # Create a session with connection pooling
    session = requests.Session()
    session.headers.update({'User-Agent': 'TrialPanorama-Processor/1.0'})
    
    # Configure session for better performance
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3,
        pool_block=False
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    results = []
    failed = []
    
    # Track progress
    start_time = time.time()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_nct = {
            executor.submit(fetch_ctgov_sync, session, nct_id): nct_id 
            for nct_id in nct_ids
        }
        
        # Create progress bar
        pbar = tqdm(total=len(nct_ids), desc="Fetching CT.gov (threaded)", unit="req")
        
        # Process results as they complete
        for future in as_completed(future_to_nct):
            try:
                nct_id, data, error = future.result()
                if data:
                    results.append((nct_id, data))
                else:
                    failed.append((nct_id, error))
            except Exception as e:
                nct_id = future_to_nct[future]
                failed.append((nct_id, str(e)))
            
            completed += 1
            pbar.update(1)
            
            # Update progress every 50 requests
            if completed % 50 == 0 or completed == len(nct_ids):
                log_retrieval_progress(completed, len(nct_ids), start_time, "threaded_fetch")
                current_rate = tracking["retrieval_stats"]["threaded_fetch"]["rate_per_second"]
                pbar.set_postfix({"rate": f"{current_rate:.1f}/s", "success": len(results), "failed": len(failed)})
        
        pbar.close()
    
    session.close()
    return results, failed

# ---------- Choose the best method ----------
start_stage("ctgov_retrieval", "Fetching study data from ClinicalTrials.gov API")

unique_nct_ids = bijective_links["nct_id"].unique()
print(f"Fetching data from ClinicalTrials.gov for {len(unique_nct_ids):,} unique NCT IDs...")

# Try async first, fallback to threaded if async fails
try:
    print("Attempting async fetch...")
    start_time = time.time()
    results, failed = asyncio.run(fetch_all_async(unique_nct_ids, max_concurrent=50))
    async_time = time.time() - start_time
    print(f"Async fetch completed in {async_time:.2f} seconds ({len(unique_nct_ids)/async_time:.1f} requests/sec)")
    
    # Log final retrieval stats
    tracking["retrieval_stats"]["final"] = {
        "method": "async",
        "total_requests": len(unique_nct_ids),
        "successful": len(results),
        "failed": len(failed),
        "success_rate": len(results) / len(unique_nct_ids) * 100,
        "total_time": async_time,
        "rate_per_second": len(unique_nct_ids) / async_time
    }
    
except Exception as e:
    print(f"Async fetch failed ({e}), falling back to threaded approach...")
    start_time = time.time()
    results, failed = fetch_all_threaded(unique_nct_ids, max_workers=20)
    threaded_time = time.time() - start_time
    print(f"Threaded fetch completed in {threaded_time:.2f} seconds ({len(unique_nct_ids)/threaded_time:.1f} requests/sec)")
    
    # Log final retrieval stats
    tracking["retrieval_stats"]["final"] = {
        "method": "threaded",
        "total_requests": len(unique_nct_ids),
        "successful": len(results),
        "failed": len(failed),
        "success_rate": len(results) / len(unique_nct_ids) * 100,
        "total_time": threaded_time,
        "rate_per_second": len(unique_nct_ids) / threaded_time
    }

print(f"Successfully fetched {len(results):,} studies from ClinicalTrials.gov")
if failed:
    print(f"Failed to fetch {len(failed):,} studies")
    print("First 5 failures:", [f"{nct}: {error}" for nct, error in failed[:5]])

print_current_status()
end_stage()

# ---------- 7) Process the fetched data ----------
start_stage("data_processing", "Parsing and structuring fetched CT.gov data")

def extract_study_data(nct_id, raw_data):
    """Extract structured data from ClinicalTrials.gov API response"""
    record = {"nct_id": nct_id, "raw_data": raw_data}
    
    try:
        protocol = raw_data.get("protocolSection", {})
        status_module = protocol.get("statusModule", {})
        identification_module = protocol.get("identificationModule", {})
        design_module = protocol.get("designModule", {})
        
        # Extract structured data
        record.update({
            "brief_title": identification_module.get("briefTitle"),
            "official_title": identification_module.get("officialTitle"),
            "overall_status": status_module.get("overallStatus"),
            "phase": design_module.get("phases", [None])[0] if design_module.get("phases") else None,
            "study_type": design_module.get("studyType"),
        })
        
        # Extract dates
        def extract_date(date_struct):
            if isinstance(date_struct, dict):
                return date_struct.get("date")
            return None
        
        record.update({
            "start_date": extract_date(status_module.get("startDateStruct")),
            "primary_completion_date": extract_date(status_module.get("primaryCompletionDateStruct")),
            "completion_date": extract_date(status_module.get("completionDateStruct")),
            "first_posted_date": extract_date(status_module.get("studyFirstPostDateStruct")),
            "last_update_date": extract_date(status_module.get("lastUpdatePostDateStruct")),
        })
        
    except Exception as e:
        print(f"Error parsing data for NCT {nct_id}: {e}")
    
    return record

# Process all results in parallel
print("Processing fetched data...")
with ThreadPoolExecutor(max_workers=4) as executor:
    ctgov_records = list(tqdm(
        executor.map(lambda x: extract_study_data(x[0], x[1]), results),
        total=len(results),
        desc="Processing data"
    ))

end_stage()

# ---------- 8) Create final datasets ----------
start_stage("dataset_creation", "Creating final output datasets")

df_ctgov = pd.DataFrame(ctgov_records)

# Create summary dataset without raw JSON
if not df_ctgov.empty:
    df_ctgov_summary = df_ctgov.drop(columns=["raw_data"], errors="ignore")
else:
    df_ctgov_summary = pd.DataFrame()

end_stage()

# ---------- 9) Save outputs ----------
start_stage("output_saving", "Saving processed data to files")

# Calculate final statistics
total_elapsed = time.time() - tracking["start_time"]

# Save bijective links
bijective_links.to_csv("study_to_nct_links_bijective.csv", index=False)
print(f"Saved bijective links: {len(bijective_links):,} rows")

# Save full ClinicalTrials.gov data (with raw JSON)
if not df_ctgov.empty:
    df_ctgov.to_json("ctgov_studies_full.jsonl", orient="records", lines=True)
    print(f"Saved full CT.gov data: {len(df_ctgov):,} studies")

# Save summary data (without raw JSON)
if not df_ctgov_summary.empty:
    df_ctgov_summary.to_csv("ctgov_studies_summary.csv", index=False)
    print(f"Saved CT.gov summary: {len(df_ctgov_summary):,} studies")

# Save comprehensive processing statistics
failed_ncts = [nct for nct, error in failed]
stats = {
    "processing_summary": {
        "total_runtime_seconds": total_elapsed,
        "total_runtime_minutes": total_elapsed / 60,
        "start_time": datetime.fromtimestamp(tracking["start_time"]).isoformat(),
        "end_time": datetime.now().isoformat(),
    },
    "stage_timings": {
        stage: {
            "duration_seconds": data["duration"],
            "description": data["description"]
        } for stage, data in tracking["stages"].items() if data["duration"] is not None
    },
    "data_flow": {
        "initial_studies_loaded": len(df_studies) + sum(s["filtered_out"] for s in tracking["filtering_stats"].values() if "study" in s.get("description", "").lower()),
        "initial_relations_loaded": len(df_rel) + sum(s["filtered_out"] for s in tracking["filtering_stats"].values() if "relation" in s.get("description", "").lower()),
        "final_studies_after_dedup": len(df_studies),
        "final_relations_after_dedup": len(df_rel),
        "direct_nct_studies": len(direct_nct),
        "indirect_study_nct_mappings": len(indirect_map) if not indirect_map.empty else 0,
        "total_mappings_before_bijection": len(all_links),
        "bijective_mappings": len(bijective_links),
        "unique_nct_ids_to_fetch": len(unique_nct_ids),
        "successful_ctgov_fetches": len(results),
        "failed_ctgov_fetches": len(failed),
        "final_processed_studies": len(ctgov_records)
    },
    "filtering_details": tracking["filtering_stats"],
    "retrieval_performance": tracking["retrieval_stats"],
    "success_rates": {
        "study_deduplication_retention": tracking["filtering_stats"].get("study_deduplication", {}).get("retention_rate", 0),
        "bijection_retention": tracking["filtering_stats"].get("bijection_filtering", {}).get("retention_rate", 0),
        "ctgov_fetch_success": (len(results) / len(unique_nct_ids) * 100) if len(unique_nct_ids) > 0 else 0,
    }
}

with open("comprehensive_processing_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# Save failed NCTs for debugging
if failed:
    failed_df = pd.DataFrame(failed, columns=["nct_id", "error"])
    failed_df.to_csv("failed_nct_ids.csv", index=False)

end_stage()

# ---------- Final Summary ----------
print("\n" + "="*60)
print("🎉 PROCESSING COMPLETE")
print("="*60)

print(f"\n⏱️  PERFORMANCE SUMMARY:")
print(f"   Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f} seconds)")
if tracking["retrieval_stats"].get("final"):
    final_stats = tracking["retrieval_stats"]["final"]
    print(f"   Retrieval method: {final_stats['method']}")
    print(f"   Retrieval speed: {final_stats['rate_per_second']:.1f} requests/second")
    print(f"   Retrieval success rate: {final_stats['success_rate']:.1f}%")

print(f"\n📊 DATA FLOW SUMMARY:")
studies_initial = stats["data_flow"]["initial_studies_loaded"]
studies_final = stats["data_flow"]["final_processed_studies"]
overall_retention = (studies_final / studies_initial * 100) if studies_initial > 0 else 0

print(f"   Studies: {studies_initial:,} → {studies_final:,} ({overall_retention:.1f}% end-to-end retention)")
print(f"   Relations: {stats['data_flow']['initial_relations_loaded']:,} → {stats['data_flow']['final_relations_after_dedup']:,}")
print(f"   NCT mappings: {stats['data_flow']['bijective_mappings']:,} bijective links created")

print(f"\n📁 OUTPUT FILES:")
print(f"   📄 study_to_nct_links_bijective.csv ({len(bijective_links):,} rows)")
print(f"   📄 ctgov_studies_full.jsonl ({len(ctgov_records):,} studies with full data)")
print(f"   📄 ctgov_studies_summary.csv ({len(df_ctgov_summary):,} studies, structured fields only)")
print(f"   📄 comprehensive_processing_stats.json (detailed metrics)")
if failed:
    print(f"   📄 failed_nct_ids.csv ({len(failed):,} failed requests)")

print(f"\n🔍 KEY FILTERING STEPS:")
for step_name, step_data in tracking["filtering_stats"].items():
    retention = step_data["retention_rate"]
    filtered = step_data["filtered_out"]
    print(f"   {step_name}: {filtered:,} filtered out ({100-retention:.1f}% reduction)")

print(f"\n✅ Processing completed successfully!")
print(f"   Check comprehensive_processing_stats.json for detailed metrics")
print("="*60)
