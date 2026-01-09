#!/usr/bin/env python3
"""Upload proof_of_time dataset to HuggingFace Hub."""

from huggingface_hub import HfApi, login
from pathlib import Path

def upload_dataset(repo_id: str, token: str = None):
    """
    Upload dataset to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "AIM-Harvard/proof-of-time")
        token: HuggingFace API token (or set HF_TOKEN env var)
    """
    # Login to HuggingFace
    if token:
        login(token=token)
    else:
        login()  # Uses HF_TOKEN from environment

    # Initialize API
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True
    )

    # Upload entire directory
    print(f"Uploading dataset to {repo_id}...")
    api.upload_folder(
        folder_path="huggingface_data",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial dataset upload (Tiers 1-2: benchmarks + sandbox data)"
    )

    print(f"✓ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g., AIM-Harvard/proof-of-time)")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    upload_dataset(args.repo_id, args.token)
