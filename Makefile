.PHONY: install test clean plots report benchmarks help ablations upload-hf prepare-hf setup dashboard

# Default target
help:
	@echo "Proof of Time - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies via uv and Inspect AI"
	@echo "  make setup            Full setup: install + configure .env"
	@echo ""
	@echo "Running Benchmarks:"
	@echo "  make test             Quick test run (5 samples, award benchmark)"
	@echo "  make benchmarks       Run all award prediction benchmarks"
	@echo "  make ablations        Run systematic ablations across models"
	@echo "  make message-sweep    Run message limit ablation sweep (15, 30, 50)"
	@echo ""
	@echo "Analysis:"
	@echo "  make analysis         Run full analysis pipeline"
	@echo "  make plots            Generate plots only"
	@echo "  make report           Generate report only"
	@echo "  make dashboard        Launch interactive EMNLP dashboard"
	@echo ""
	@echo "Dataset:"
	@echo "  make prepare-hf       Prepare HuggingFace dataset structure"
	@echo "  make upload-hf        Upload dataset to HuggingFace Hub"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove generated files (preserve logs)"
	@echo "  make clean-all        Remove all generated files and logs"
	@echo ""

# Installation
install:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "Installing Inspect AI..."
	pip install inspect-ai
	@echo "✓ Installation complete"

setup: install
	@echo "Creating .env file from template..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "✓ Created .env - please edit with your API keys"; else echo "⚠ .env already exists"; fi
	@echo "To activate environment variables, run: export \$$(cat .env | xargs)"

# Testing
test:
	@echo "Running quick test (5 samples)..."
	inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_simple_task \
		--model openai/gpt-5-mini-2025-08-07 \
		--limit 5

benchmarks:
	@echo "Running all award prediction benchmarks..."
	inspect eval benchmarks/award_react/benchmark.py \
		--model openai/gpt-5-mini-2025-08-07

ablations:
	@echo "Running systematic ablations across models..."
	uv run scripts/run_inspect_ablations.py

message-sweep:
	@echo "Running message limit ablation sweep..."
	bash run_message_limit_sweep.sh

# Analysis
analysis:
	@echo "Running full analysis pipeline..."
	cd analysis/comprehensive && python main.py

plots:
	@echo "Generating plots..."
	cd analysis/comprehensive && python main.py --plots-only

report:
	@echo "Generating report..."
	cd analysis/comprehensive && python main.py --report-only

dashboard:
	@echo "Launching EMNLP topics dashboard..."
	@echo "Access at http://127.0.0.1:8050"
	uv sync --extra dashboard
	python analysis/emnlp_topics_dashboard.py

# HuggingFace Dataset
prepare-hf:
	@echo "Preparing HuggingFace dataset structure..."
	@mkdir -p huggingface_data/{benchmarks,sandbox_data}/{award_react,citation_react,future_work_react,sota_forecast}
	@mkdir -p huggingface_data/sandbox_data/{citation,award,faculty,sota}
	@echo "Copying benchmark JSONL files (Tier 1)..."
	@cp benchmarks/award_react/*.jsonl huggingface_data/benchmarks/award_react/ 2>/dev/null || true
	@cp benchmarks/citation_react/*.jsonl huggingface_data/benchmarks/citation_react/ 2>/dev/null || true
	@cp benchmarks/future_work_react/*.jsonl huggingface_data/benchmarks/future_work_react/ 2>/dev/null || true
	@cp benchmarks/sota_forecast/mcq_dataset.jsonl huggingface_data/benchmarks/sota_forecast/ 2>/dev/null || true
	@echo "Copying sandbox data (Tier 2)..."
	@cp benchmarks/citation_react/sandbox/data/historical_papers_2021_2024.jsonl huggingface_data/sandbox_data/citation/ 2>/dev/null || true
	@cp benchmarks/award_react/sandbox/data/accepted_papers.csv huggingface_data/sandbox_data/award/ 2>/dev/null || true
	@cp benchmarks/future_work_react/sandbox/data/faculty_publications.jsonl huggingface_data/sandbox_data/faculty/ 2>/dev/null || true
	@cp benchmarks/sota_forecast/sandbox/data/sota_metrics.json huggingface_data/sandbox_data/sota/ 2>/dev/null || true
	@if [ -d benchmarks/future_work_react/sandbox/data/faculty_publications ]; then \
		echo "Compressing faculty CSV files..."; \
		tar -czf huggingface_data/sandbox_data/faculty/faculty_publications.tar.gz \
			-C benchmarks/future_work_react/sandbox/data/faculty_publications .; \
	fi
	@echo "✓ HuggingFace data structure prepared in huggingface_data/"
	@du -sh huggingface_data/* 2>/dev/null || true

upload-hf:
	@echo "Uploading dataset to HuggingFace Hub..."
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "⚠ HF_TOKEN not set. Please set it or run: huggingface-cli login"; \
		exit 1; \
	fi
	python scripts/upload_to_huggingface.py AIM-Harvard/proof-of-time
	@echo "✓ Dataset uploaded to https://huggingface.co/datasets/AIM-Harvard/proof-of-time"

# Cleaning
clean:
	@echo "Cleaning generated files..."
	rm -rf analysis/comprehensive/plots/
	rm -f analysis/comprehensive/REPORT.md
	rm -rf analysis/comprehensive/tables/
	@echo "✓ Cleaned generated files (logs preserved)"

clean-all: clean
	@echo "Removing all logs..."
	rm -rf logs/
	rm -rf huggingface_data/
	@echo "✓ Cleaned all generated files and logs"
