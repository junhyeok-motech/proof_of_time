# Proof of Time: Benchmarking LLM Agents on Academic Paper Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/AIM-Harvard/proof-of-time)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Proof of Time** is a benchmark suite for evaluating LLM agents on academic paper analysis tasks that require understanding research trends, citations, and future directions. All tasks use post-training-cutoff data to avoid data contamination.

**Paper**: *Proof of Time: Benchmarking LLM Agents on Academic Paper Analysis* (Under Review)

**Key Features:**
- 4 benchmark families with 10+ evaluation tasks
- Post-training-cutoff datasets (2025 papers)
- ReAct agent framework with sandboxed paper data
- Comprehensive analysis pipeline for results
- Support for OpenAI, Google Gemini, and Anthropic Claude models

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/shan23chen/proof_of_time.git
cd proof_of_time
uv sync
pip install inspect-ai

# 2. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Run a quick test (5 samples)
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_simple_task \
    --model openai/gpt-5-mini-2025-08-07 \
    --limit 5
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Table of Contents

- [Overview](#overview)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Running Benchmarks](#running-benchmarks)
- [Dataset](#dataset)
- [Analysis](#analysis)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Overview

Proof of Time evaluates LLM agents on tasks that require understanding research trends and making predictions about academic papers. All evaluation data is from **post-training-cutoff** (2025), ensuring models cannot rely on memorized information.

### Why "Proof of Time"?

The name reflects our focus on temporal reasoning: agents must analyze historical patterns to make predictions about future research directions, award recipients, and citation impact. Tasks require genuine understanding of research trends rather than memorization.

### Benchmark Design

- **ReAct Agents**: Agents use tools (bash, Python, text editor) to explore sandboxed paper datasets
- **Sandboxed Environments**: Docker containers with read-only paper data (no internet access)
- **Offline Prompt**: Custom "Antigravity" prompt inspired by principles of focused exploration
- **Multiple Variants**: Each task has standard (agent), simple (zero-shot), and no-offline-prompt versions

## Benchmarks

The suite includes 4 benchmark families:

### 1. Award Prediction ([benchmarks/award_react/](benchmarks/award_react/))

Predict which papers will win best paper awards at top NLP conferences.

**Tasks:**
- Pre-cutoff awards (ACL/EMNLP/NAACL 2018-2024)
- Post-cutoff EMNLP 2025 awards
- Post-cutoff ACL/NAACL 2025 awards

**Agent Access:** CSV of accepted papers with titles, authors, abstracts

**Example:** Given 5 EMNLP 2025 papers, which won the best paper award?

### 2. Citation Forecasting ([benchmarks/citation_react/](benchmarks/citation_react/))

Predict future citation counts for recently published papers.

**Tasks:**
- Multiple choice: Select highest-cited paper
- Ranking: Rank papers by predicted citations
- Bucket prediction: Classify into citation ranges (0-1, 1-5, 5-10, 10-50, 50+)

**Agent Access:** Historical papers (2021-2024) with citation counts

**Example:** Which of these 2025 papers will have the most citations by 2026?

### 3. Faculty Future Work ([benchmarks/future_work_react/](benchmarks/future_work_react/))

Predict research directions of AI faculty members based on publication history.

**Tasks:**
- Professor field: Predict research field for future work
- Professor article: Predict which paper a professor would author
- Field focus: Classify focus area within a field

**Agent Access:** Per-professor CSV files with publication history

**Example:** Based on their publications, will this professor's next paper be in NLP, CV, or RL?

### 4. SOTA Forecasting ([benchmarks/sota_forecast/](benchmarks/sota_forecast/))

Predict state-of-the-art performance ranges on ML benchmarks.

**Tasks:**
- Bucket prediction: Classify SOTA scores into ranges (0-20, 20-40, 40-60, 60-80, 80-100)

**Agent Access:** JSON table of October 2025 frontier model scores

**Example:** What performance bucket does the best model achieve on MMLU in Oct 2025?

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker (for sandbox environments)
- uv package manager
- API keys for LLM providers

### Quick Install

```bash
# Clone repository
git clone https://github.com/shan23chen/proof_of_time.git
cd proof_of_time

# Install dependencies
uv sync
pip install inspect-ai

# Configure API keys
cp .env.example .env
# Edit .env with your OpenAI/Anthropic/Google API keys
export $(cat .env | xargs)

# Verify Docker is running
docker ps
```

For detailed installation instructions, troubleshooting, and Docker setup, see [SETUP.md](SETUP.md).

## Running Benchmarks

### Single Task Example

```bash
# Run award prediction with GPT-5-mini (5 samples)
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_task \
    --model openai/gpt-5-mini-2025-08-07 \
    --limit 5
```

### All Tasks for a Benchmark

```bash
# Run all award prediction variants
inspect eval benchmarks/award_react/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07

# Run all citation forecasting variants
inspect eval benchmarks/citation_react/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07
```

### Systematic Ablations

Run all benchmarks across multiple models with ablations:

```bash
# Run all models and tasks (with/without offline prompt)
uv run scripts/run_inspect_ablations.py

# Run specific models only
uv run scripts/run_inspect_ablations.py \
    --models openai/gpt-5-mini-2025-08-07 google/gemini-3-flash-preview

# Run with different message limits
bash run_message_limit_sweep.sh
```

**Supported Models:**
- OpenAI: `gpt-5.2`, `gpt-5.1`, `gpt-5-mini`, `gpt-5-nano`
- Google: `gemini-3-pro`, `gemini-3-flash`, `vertex/gemini-2.5-pro/flash`
- Anthropic: `vertex/claude-opus-4-5`, `vertex/claude-sonnet-4-5`, `vertex/claude-haiku-4-5`

Logs are saved to `logs/ablations/<model-slug>/<task>.log`.

## Dataset

### HuggingFace Hub

Benchmark datasets and sandbox data are available on HuggingFace:

**Repository:** [AIM-Harvard/proof-of-time](https://huggingface.co/datasets/AIM-Harvard/proof-of-time)

```bash
# Download with datasets library
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('AIM-Harvard/proof-of-time')"

# Or clone manually
git clone https://huggingface.co/datasets/AIM-Harvard/proof-of-time
```

### Dataset Structure

```
benchmarks/
├── award_react/          # 641 KB - Award prediction tasks
├── citation_react/       # 2.7 MB - Citation forecasting tasks
├── future_work_react/    # 469 KB - Faculty prediction tasks
└── sota_forecast/        # 26 KB - SOTA benchmark tasks

sandbox_data/
├── citation/             # 21 MB - Historical papers with citations
├── award/                # 19 MB - Conference accepted papers
├── faculty/              # 38 MB - Faculty publication histories
└── sota/                 # 8.7 KB - Frontier model benchmark scores
```

### Regenerating Datasets

To create custom datasets or update with new data:

```bash
# Regenerate award prediction tasks
uv run python dataset_building/generate_award_datasets.py

# Regenerate citation forecasting tasks
uv run python dataset_building/generate_citation_datasets.py

# Regenerate faculty prediction tasks
uv run python dataset_building/generate_faculty_futurework.py

# Regenerate SOTA forecasting tasks
uv run python dataset_building/generate_sota_forecast.py
```

See [dataset_building/README.md](dataset_building/README.md) for detailed instructions.

## Analysis

### Comprehensive Analysis Pipeline

Generate plots, tables, and reports from evaluation logs:

```bash
# Run full analysis pipeline
cd analysis/comprehensive
python main.py

# Generate plots only
python main.py --plots-only

# Generate report only
python main.py --report-only
```

**Outputs:**
- `plots/` - Performance visualizations by model, task, ablation
- `tables/` - CSV summaries of results
- `REPORT.md` - Comprehensive markdown report

### Interactive Dashboard

Explore EMNLP paper topics and trends:

```bash
# Install dashboard dependencies
uv sync --extra dashboard

# Launch dashboard
python analysis/emnlp_topics_dashboard.py

# Export CSV summaries without UI
python analysis/emnlp_topics_dashboard.py --export
```

Access at http://127.0.0.1:8050 (use `--host` and `--port` to customize).

See [analysis/README.md](analysis/README.md) for more details.

## Repository Structure

```
proof_of_time/
├── benchmarks/              # Inspect AI benchmark implementations
│   ├── award_react/         # Award prediction tasks
│   ├── citation_react/      # Citation forecasting tasks
│   ├── future_work_react/   # Faculty prediction tasks
│   ├── sota_forecast/       # SOTA forecasting tasks
│   └── common/              # Shared utilities and prompts
├── dataset_building/        # Scripts to generate benchmark datasets
├── analysis/                # Result analysis and visualization
│   ├── comprehensive/       # Main analysis pipeline
│   ├── agent_behavior/      # Agent action analysis
│   └── query_analysis/      # Query pattern analysis
├── scripts/                 # Convenience scripts for running experiments
│   ├── run_inspect_ablations.py       # Systematic ablation runner
│   ├── upload_to_huggingface.py       # Dataset upload script
│   └── parse_ablation_logs.py         # Log parsing utilities
├── run_message_limit_sweep.sh         # Message limit experiments
├── .env.example             # API key configuration template
├── pyproject.toml           # Python dependencies
├── SETUP.md                 # Detailed setup instructions
├── CITATION.md              # Citation information
└── LICENSE                  # MIT License
```

## Citation

If you use this benchmark suite in your research, please cite our work:

```bibtex
@article{proof-of-time-2025,
  title={Proof of Time: Benchmarking LLM Agents on Academic Paper Analysis},
  author={TBD},
  journal={Under Review},
  year={2025}
}
```

For the dataset:

```bibtex
@dataset{proof-of-time-dataset-2025,
  title={Proof of Time: Academic Paper Analysis Benchmarks},
  author={AIM Harvard},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/AIM-Harvard/proof-of-time}
}
```

See [CITATION.md](CITATION.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! To add new benchmarks or improve existing ones:

1. Fork the repository
2. Create a feature branch
3. Add your benchmark following the patterns in `benchmarks/`
4. Add documentation (README.md for your benchmark)
5. Test with `inspect eval` and `--limit 5`
6. Submit a pull request

## Acknowledgments

- Built with [Inspect AI](https://inspect.ai-safety-institute.org.uk/) framework
- Uses data from ACL Anthology, Google Scholar, and Papers with Code
- Thanks to the AI research community for making papers publicly available

## Contact

- **Issues**: https://github.com/shan23chen/proof_of_time/issues
- **Email**: aim@seas.harvard.edu
- **Project**: AIM Harvard
