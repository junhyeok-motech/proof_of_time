# EMNLP React Benchmark Prototype

This prototype wires up an [Inspect AI](https://inspect.aisi.org.uk/) benchmark that evaluates a simple React agent confined to a local sandbox of EMNLP accepted papers.

## Layout

- `benchmark.py` – benchmark entry point (`emnlp_react_task`) registering the dataset, sandbox, agent, and scorer.
- `qa_dataset.jsonl` – three demonstration QA prompts plus string targets.
- `sandbox/` – workspace mounted for the agent. Contains a flattened JSONL corpus (`data/emnlp_papers.jsonl`) and the original zip export (`datasets/emnlp_with_topics_benchmarks.zip`).

## Quickstart

1. Install Inspect AI into your environment (for example: `pip install inspect-ai`). The package is **not** vendored in this repository.
2. From the repository root, run:
   ```bash
   inspect eval inspect/emnlp_react/benchmark.py::emnlp_react_task
   ```
3. Inspect AI will spin up the React agent with the standard tool suite (think, python, bash, bash session, text editor) restricted to the sandbox directory. The agent must read the JSONL corpus to answer each QA item.

You can expand the QA dataset or build new tasks by editing `qa_dataset.jsonl`, regenerating the JSONL corpus in `sandbox/data/`, or enriching the sandbox with additional derived artefacts.
