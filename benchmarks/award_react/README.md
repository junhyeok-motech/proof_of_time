# *CL Conference Award Tier Classification Benchmark - Inspect AI

This benchmark evaluates LLM agents on classifying NLP conference papers into award tiers using a sandboxed environment with accepted-paper data.

## Overview

The agent must assign one of four recognition tiers using only sandbox data:

- **Best**: Best Paper Award winners (paradigm-shifting work)
- **Outstanding**: Outstanding Paper Awards (exceptional contributions)
- **Main**: Main conference track (accepted papers)
- **Findings**: Findings track (good work, below main bar)

The sandbox provides accepted-paper metadata so the agent can compare a target paper to historical patterns and award winners.

You can download our generated questions from: https://huggingface.co/datasets/AIM-Harvard/proof-of-time/tree/main/benchmarks

## Tasks

### 1. Pre-Cutoff Tasks (`pre_cutoff_task`)

**Purpose**: Evaluate on historical papers (ACL/EMNLP/NAACL 2018-2024) that may be in model training data.

**Question**: Which recognition tier best fits this paper?

**Example**:

```
Title: The Paper Title
Abstract: The abstract text...
Authors: Author A; Author B

Question: Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?
Options: Findings, Main, Outstanding, Best

Answer with only one word: Best, Outstanding, Main, or Findings.
```

**Dataset**: `pre-cutoff_mcq.jsonl`

### 2. Post-Cutoff EMNLP Tasks (`post_cutoff_emnlp_task`)

**Purpose**: Evaluate on 2025+ EMNLP papers that are after model knowledge cutoff (temporal contamination-free).

**Dataset**: `post-cutoff_emnlp.jsonl`

### 3. Post-Cutoff ACL/NAACL Tasks (`post_cutoff_acl_naacl_task`)

**Purpose**: Evaluate on 2025+ ACL/NAACL papers that are after model knowledge cutoff.

**Dataset**: `post-cutoff_acl_naacl.jsonl`

### 4. Simple Baselines

No-tools baselines that answer directly without sandbox inspection:
- `pre_cutoff_simple_task`
- `post_cutoff_emnlp_simple_task`
- `post_cutoff_acl_naacl_simple_task`

### Variants

Each core task also has:

- **`_no_offline_prompt`** variants (without the shared offline preamble)
- **`_local`** variants (no Docker sandbox; direct file access)

## Running the Benchmark

### Prerequisites

```bash
# Install Inspect AI
pip install inspect-ai

# Ensure Docker is running
docker ps
```

### Run Individual Tasks

```bash
# Pre-cutoff (historical papers)
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_task \
  --model openai/gpt-4o-mini

# Post-cutoff EMNLP (2025+ papers)
inspect eval benchmarks/award_react/benchmark.py@post_cutoff_emnlp_task \
  --model openai/gpt-4o-mini

# Post-cutoff ACL/NAACL (2025+ papers)
inspect eval benchmarks/award_react/benchmark.py@post_cutoff_acl_naacl_task \
  --model openai/gpt-4o-mini

# Simple baseline (no tools)
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_simple_task \
  --model openai/gpt-4o-mini
```

### With Different Models

```bash
# Claude
inspect eval benchmarks/award_react/benchmark.py@post_cutoff_emnlp_task \
  --model anthropic/claude-3-5-sonnet-20241022

# GPT-4
inspect eval benchmarks/award_react/benchmark.py@post_cutoff_emnlp_task \
  --model openai/gpt-4o

# Limit samples for quick testing
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_task \
  --model openai/gpt-4o-mini \
  --limit 5

# Local variant (no Docker)
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_task_local \
  --model openai/gpt-4o-mini
```

## Sandbox Environment

### Data Files

Expected in `sandbox/data/` (mounted read-only inside the Docker sandbox):

- `accepted_papers.csv`: accepted-paper corpus with titles, abstracts, authors, venues, award status

### Task Files

Located in `benchmarks/award_react/`:

- `pre-cutoff_mcq.jsonl`: Pre-2025 conference awards (ACL/EMNLP/NAACL 2018-2024)
- `post-cutoff_emnlp.jsonl`: Post-2025 EMNLP awards
- `post-cutoff_acl_naacl.jsonl`: Post-2025 ACL/NAACL awards

### Available Tools

- `bash()`: Execute shell commands
- `python()`: Run Python code for analysis
- `bash_session()`: Persistent shell session
- `text_editor()`: Read/write files
- `think()`: Chain-of-thought reasoning

### Network Access

**Disabled** - agent must use only sandbox data.

## Agent Strategy

The React agent is prompted to:

1. Load the accepted-paper corpus from `sandbox/data/accepted_papers.csv`
2. Inspect Best/Outstanding examples to learn award patterns
3. Compare the target paper's novelty, impact, and quality to those examples
4. Output a single-word tier label (Best, Outstanding, Main, Findings)

## Evaluation Metrics

### Accuracy

- **All tasks**: Exact match on the tier label (Best, Outstanding, Main, or Findings)

## Data Sources

Generated from:

- Award spreadsheet(s) of Best/Outstanding winners
- Accepted-papers corpora (ACL/EMNLP/NAACL)
- Dataset building scripts in `dataset_building/` (see `dataset_building/README.md`)

## Troubleshooting

### Missing dataset files

If `accepted_papers.csv` or the JSONL task files are missing, regenerate them using the dataset building scripts referenced in `dataset_building/README.md`.

### Docker issues

```bash
# Check Docker is running
docker ps

# Pull Python image
docker pull python:3.11-slim
```

## License

MIT License - see main repository LICENSE file.
