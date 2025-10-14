import os
import json
from typing import List, Dict, Any

from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi


DATASET_IDS = [
    "AIM-Harvard/EMNLP-Accepted-Papers",
    "AIM-Harvard/ACL-Accepted-Papers",
    "AIM-Harvard/NAACL-Accepted-Papers",
    "AIM-Harvard/NIPS-Accepted-Papers",
    "AIM-Harvard/ICML-Accepted-Papers",
    "AIM-Harvard/ICLR-Accepted-Papers",
]


def iter_all_examples(dataset_id: str):
    """Yield all examples from every config and split for a dataset id.

    Adds source metadata fields for provenance.
    """
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        configs = [None]

    for config in configs:
        try:
            splits = get_dataset_split_names(dataset_id, config) if config else get_dataset_split_names(dataset_id)
        except Exception:
            # Fall back to common default split
            splits = ["train"]

        for split in splits:
            try:
                ds = load_dataset(dataset_id, config, split=split, streaming=True) if config else load_dataset(dataset_id, split=split, streaming=True)
            except Exception as e:
                # Try non-streaming as fallback
                try:
                    ds = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
                except Exception:
                    print(f"[skip] {dataset_id} config={config} split={split}: {e}")
                    continue

            count = 0
            for ex in ds:
                # Ensure JSON-serializable: convert non-serializable types if any
                if isinstance(ex, dict):
                    record = dict(ex)
                else:
                    try:
                        record = ex.to_dict()  # type: ignore[attr-defined]
                    except Exception:
                        record = json.loads(json.dumps(ex, default=str))
                record["source_dataset"] = dataset_id
                if config is not None:
                    record["source_config"] = str(config)
                record["source_split"] = str(split)
                yield record
                count += 1
                if count % 5000 == 0:
                    print(f"[progress] {dataset_id}::{config or 'default'}::{split} -> {count} examples")


def write_jsonl(records, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if total % 10000 == 0:
                print(f"[write] wrote {total} records to {out_path}")
    print(f"[done] wrote {total} total records to {out_path}")


def try_upload_to_hf(local_path: str, repo_id: str | None = None, repo_namespace: str | None = None) -> str | None:
    """Attempt to create/update a dataset repo and upload the JSONL file.

    Returns the hf hub file url on success, else None.
    """
    api = HfApi()
    try:
        who = api.whoami()
        user = who.get("name") or who.get("fullname") or ""
    except Exception as e:
        print(f"[upload] Skipping upload: not logged in ({e})")
        return None

    # Determine target repo id
    if not repo_id:
        namespace = repo_namespace or user
        repo_id = f"{namespace}/Accepted-Papers-Aggregated"

    print(f"[upload] Target dataset repo: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"[upload] create_repo failed: {e}")

    # Upload file under data/
    remote_path = "data/accepted_papers.jsonl"
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=remote_path,
        )
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"
        print(f"[upload] Uploaded to {url}")
        return url
    except Exception as e:
        print(f"[upload] Upload failed: {e}")
        return None


def main():
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "inspect",
        "emnlp_react",
        "sandbox",
        "data",
        "emnlp_papers.jsonl",
    )

    print("[start] aggregating datasets:")
    for ds in DATASET_IDS:
        print(f" - {ds}")

    # Aggregate
    all_records = iter(
        rec for dataset_id in DATASET_IDS for rec in iter_all_examples(dataset_id)
    )
    write_jsonl(all_records, out_path)

    # Try to upload back to HF (if logged in)
    # Optional overrides for upload destination
    repo_id_override = os.environ.get("HF_DATASET_REPO")  # full repo id e.g. org/name
    repo_ns_override = os.environ.get("HF_DATASET_NAMESPACE")  # namespace/user or org
    try_upload_to_hf(out_path, repo_id=repo_id_override, repo_namespace=repo_ns_override)


if __name__ == "__main__":
    main()
