# Analysis Pipeline

This directory contains tools for analyzing Proof of Time benchmark results.

## Directory Structure

```
analysis/
├── comprehensive/           # Main analysis pipeline
│   ├── main.py              # Entry point for full analysis
│   ├── data_loader.py       # Load and preprocess evaluation CSVs
│   ├── plots.py             # Plot generation functions
│   ├── generate_plots.py    # Plot orchestration
│   ├── report_generator.py  # Markdown report generation
│   ├── plots/               # Generated figures (PNG + PDF)
│   └── REPORT.md            # Generated analysis report
│
├── agent_behavior/          # Agent execution analysis
│   ├── sample_task_examples_stratified.py  # Stratified sampling
│   ├── inputs/              # Input data directory
│   ├── outputs/             # Generated analysis outputs
│   └── README.md            # Detailed documentation
│
├── emnlp_topics_dashboard.py    # EMNLP paper topic explorer
├── future_work_dashboard.py     # Faculty prediction explorer
│
└── README.md                # This file
```

## Quick Start

### Comprehensive Analysis Pipeline

Generate publication-ready plots and reports from evaluation logs:

```bash
cd analysis/comprehensive

# Run full analysis (plots + report)
python main.py

# Generate plots only
python main.py --plots-only

# Generate report only
python main.py --report-only
```

**Outputs:**
- `plots/` - Performance visualizations (PNG and PDF formats)
- `REPORT.md` - Comprehensive markdown analysis report

### Agent Behavior Analysis

Analyze agent execution patterns with stratified sampling:

```bash
cd analysis/agent_behavior

# Sample 2 examples per stratum (default)
python sample_task_examples_stratified.py

# Sample more examples per stratum
python sample_task_examples_stratified.py --samples-per-stratum 5

# Save combined output
python sample_task_examples_stratified.py \
  --combined-output outputs/all_samples_combined.jsonl
```

**Stratification Categories:**
- **Complete + Correct**: Agent got right answer, didn't hit message limit
- **Complete + Wrong**: Agent got wrong answer, didn't hit message limit
- **Incomplete**: Agent hit the message limit

See [agent_behavior/README.md](agent_behavior/README.md) for detailed documentation.

### EMNLP Topics Dashboard

Interactive explorer for EMNLP paper topics, trends, and benchmark activity (2021-2024):

```bash
# Launch interactive dashboard
python analysis/emnlp_topics_dashboard.py

# Specify host and port
python analysis/emnlp_topics_dashboard.py --host 0.0.0.0 --port 8080

# Export CSV summaries without launching UI
python analysis/emnlp_topics_dashboard.py --export
```

**Features:**
- Filter by year, venue, topic, domain, benchmark task
- Primary topic trend visualization
- Fine-grained topic heatmaps
- Benchmark paper statistics
- Top authors and domain analysis
- Download filtered datasets as CSV

Access at http://127.0.0.1:8050 (default).

**Exported Reports** (with `--export`):
- `reports/emnlp_topics_summary.csv` - Primary topic trends
- `reports/emnlp_finetopic_summary.csv` - Fine-grained topic counts
- `reports/emnlp_benchmark_summary.csv` - Benchmark statistics
- `reports/emnlp_domain_summary.csv` - Domain distribution
- `reports/emnlp_author_summary.csv` - Author frequency

### Future Work Dashboard

Interactive explorer for AI faculty research direction predictions:

```bash
# Launch interactive dashboard
python analysis/future_work_dashboard.py

# Specify host and port
python analysis/future_work_dashboard.py --host 0.0.0.0 --port 8080

# Export CSV summaries without launching UI
python analysis/future_work_dashboard.py --export
```

**Features:**
- Filter by verification status, score range, verdict
- Precision/recall/F1 score distribution
- Keyword prediction vs actual comparison
- Field prediction accuracy analysis
- Top performers ranking
- Download filtered predictions as CSV

Access at http://127.0.0.1:8050 (default).

**Exported Reports** (with `--export`):
- `future_work_results/reports/prediction_summary.csv` - Overall prediction results
- `future_work_results/reports/keyword_analysis.csv` - Predicted vs actual keywords
- `future_work_results/reports/field_analysis.csv` - Field prediction accuracy
- `future_work_results/reports/top_performers.csv` - Top performing predictions

## Prerequisites

### Input Data

The analysis pipeline requires summary CSV files generated from Inspect AI evaluation logs:

```
logs_msg15_summary.csv   # 15 message limit results
logs_msg30_summary.csv   # 30 message limit results
logs_msg50_summary.csv   # 50 message limit results
```

These files should be in the project root directory.

### Dependencies

```bash
# Install project dependencies
uv sync

# Required packages (included in pyproject.toml):
# - pandas
# - matplotlib
# - seaborn
# - numpy

# For interactive dashboards (emnlp_topics_dashboard.py, future_work_dashboard.py):
uv sync --extra dashboard
# Or: pip install dash plotly datasets
```

## Output Examples

### Plots Generated

The comprehensive pipeline generates several plot types:

1. **Model-Task Heatmaps** - Performance across all model/task combinations
2. **Scaling Analysis** - Performance vs. message limit
3. **Agent vs Zero-shot** - Comparison of agentic vs simple baselines
4. **Task Family Comparisons** - Performance by task category

### Report Sections

The generated REPORT.md includes:

- Data summary and statistics
- Model performance rankings
- Task difficulty analysis
- Scaling behavior insights
- Key findings and recommendations

## Customization

### Adding New Plots

Extend `plots.py` with new visualization functions:

```python
def plot_custom_analysis(df, output_dir):
    """Generate custom analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting code ...
    fig.savefig(output_dir / "plots" / "custom_analysis.png", dpi=300)
```

Then register in `generate_plots.py`.

### Modifying Report Structure

Edit `report_generator.py` to customize report sections and formatting.

## Troubleshooting

### Missing CSV Files

If you see "Failed to load data", ensure summary CSV files exist:

```bash
ls -la logs_msg*_summary.csv
```

Generate them by running the log parsing scripts first.

### Plot Generation Errors

Ensure matplotlib backend is configured:

```bash
export MPLBACKEND=Agg  # For headless environments
```

## Notes

- Analysis assumes Inspect AI evaluation log format
- Plots are generated in both PNG (for web) and PDF (for papers) formats
- Stratified sampling uses fixed random seed (42) for reproducibility
