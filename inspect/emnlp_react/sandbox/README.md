# EMNLP React Agent Sandbox

This sandbox is mounted for Inspect AI agents so they can explore the EMNLP benchmark corpus without touching the open Internet.

- `data/emnlp_papers.jsonl` – flattened JSONL dump covering EMNLP 2021–2024 (main and findings) with benchmark annotations and metadata.
- `datasets/emnlp_with_topics_benchmarks.zip` – original Hugging Face export (kept for completeness if deeper feature access is needed).

Agents should rely on the provided JSONL file for quick filtering. It includes title, authors, abstract snippets, tags, primary and fine topics, domain tags, benchmark flags, benchmark names, and benchmark task lists.
