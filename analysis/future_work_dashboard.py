"""Interactive future work prediction explorer with verification metrics and exports."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

# -------------------- Paths --------------------
PREDICTIONS_PATH = Path("future_work_results/predictions_with_verification.json")
EXPORT_DIR = Path("future_work_results/reports")
SUMMARY_CSV = EXPORT_DIR / "prediction_summary.csv"
KEYWORD_CSV = EXPORT_DIR / "keyword_analysis.csv"
FIELD_CSV = EXPORT_DIR / "field_analysis.csv"
TOP_PERFORMERS_CSV = EXPORT_DIR / "top_performers.csv"

# -------------------- Styles --------------------
PAGE_STYLE = {"padding": "24px", "fontFamily": "'Helvetica Neue', Arial, sans-serif"}
FILTERS_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
    "gap": "16px",
    "margin": "24px 0",
}
FILTER_CARD_STYLE = {
    "background": "#f7f9fc",
    "border": "1px solid #dde3ea",
    "borderRadius": "8px",
    "padding": "12px",
}
GRAPH_GRID_STYLE = {"display": "grid", "gap": "24px", "margin": "24px 0"}
SUMMARY_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
    "gap": "16px",
    "marginBottom": "24px",
}
SUMMARY_CARD_STYLE = {
    "background": "#0d1b2a",
    "color": "white",
    "borderRadius": "8px",
    "padding": "16px",
    "textAlign": "center",
}

# -------------------- Data Loading --------------------
def load_predictions() -> pd.DataFrame:
    """Load predictions from JSON and convert to DataFrame."""
    with open(PREDICTIONS_PATH) as f:
        data = json.load(f)

    records = []
    for entry in data:
        # New structure: entry has professor, affiliation, result
        # result contains prediction, verification, error
        result = entry.get("result", {})
        pred = result.get("prediction", {})
        verif = result.get("verification")
        error = result.get("error")

        record = {
            "professor": entry.get("professor", pred.get("professor", "Unknown")),
            "affiliation": entry.get("affiliation", pred.get("affiliation", "")),
            "predicted_keywords": pred.get("predicted_keywords", []),
            "predicted_fields": pred.get("predicted_fields", []),
            "rationale": pred.get("rationale", ""),
            "has_verification": verif is not None,
            "error": error if error else "",
        }

        if verif and isinstance(verif, dict) and "aggregate" in verif:
            agg = verif.get("aggregate", {})
            kw_scores = agg.get("keywords", {})
            field_scores = agg.get("fields", {})

            # Extract actual keywords/fields from aggregate (new format)
            # Or from per_paper_scores if available (old format)
            actual_kw = agg.get("actual_keywords", [])
            actual_fields = agg.get("actual_fields", [])

            # Fallback to per_paper_scores if aggregate doesn't have them
            if not actual_kw:
                per_paper = verif.get("per_paper_scores", [])
                actual_kw_set = set()
                actual_fields_set = set()
                for paper in per_paper:
                    actual_kw_set.update(paper.get("actual_keywords", []))
                    actual_fields_set.update(paper.get("actual_fields", []))
                actual_kw = list(actual_kw_set)
                actual_fields = list(actual_fields_set)

            record.update({
                "keyword_precision": kw_scores.get("precision", 0.0),
                "keyword_recall": kw_scores.get("recall", 0.0),
                "keyword_f1": kw_scores.get("f1", 0.0),
                "field_precision": field_scores.get("precision", 0.0),
                "field_recall": field_scores.get("recall", 0.0),
                "field_f1": field_scores.get("f1", 0.0),
                "overall_score": agg.get("composite_f1", 0.0),
                "verdict": agg.get("verdict", "unknown"),
                "n_papers": agg.get("n_papers", 0),
                "actual_keywords": actual_kw,
                "actual_fields": actual_fields,
            })
        else:
            record.update({
                "keyword_precision": None,
                "keyword_recall": None,
                "keyword_f1": None,
                "field_precision": None,
                "field_recall": None,
                "field_f1": None,
                "overall_score": None,
                "verdict": None,
                "n_papers": 0,
                "actual_keywords": [],
                "actual_fields": [],
            })

        records.append(record)

    return pd.DataFrame(records)


def filter_predictions(
    df: pd.DataFrame,
    verification_status: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    fields: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
    search_query: str | None = None,
    verdict: str | None = None,
) -> pd.DataFrame:
    """Filter predictions based on various criteria."""
    filtered = df.copy()

    if verification_status == "verified":
        filtered = filtered[filtered["has_verification"]]
    elif verification_status == "unverified":
        filtered = filtered[~filtered["has_verification"]]

    # Score filters only apply to verified entries
    if min_score is not None:
        filtered = filtered[
            (filtered["overall_score"].isna()) | (filtered["overall_score"] >= min_score)
        ]
    if max_score is not None:
        filtered = filtered[
            (filtered["overall_score"].isna()) | (filtered["overall_score"] <= max_score)
        ]

    if fields:
        filtered = filtered[
            filtered["predicted_fields"].apply(
                lambda x: any(f in x for f in fields)
            )
        ]

    if keywords:
        filtered = filtered[
            filtered["predicted_keywords"].apply(
                lambda x: any(kw.lower() in [k.lower() for k in x] for kw in keywords)
            )
        ]

    if verdict and verdict != "all":
        filtered = filtered[filtered["verdict"] == verdict]

    if search_query:
        pattern = search_query.strip().lower()
        if pattern:
            filtered = filtered[
                filtered["professor"].str.lower().str.contains(pattern, regex=False, na=False)
                | filtered["affiliation"].str.lower().str.contains(pattern, regex=False, na=False)
            ]

    return filtered


# -------------------- Analysis Functions --------------------
def build_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build score distribution for verified predictions."""
    verified = df[df["has_verification"]].copy()
    if verified.empty:
        return pd.DataFrame()

    # Melt the precision/recall/f1 metrics
    score_types = [
        "keyword_precision", "keyword_recall", "keyword_f1",
        "field_precision", "field_recall", "field_f1",
    ]

    melted = verified.melt(
        id_vars=["professor"],
        value_vars=score_types,
        var_name="metric",
        value_name="score"
    )

    return melted


def build_keyword_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze predicted vs actual keywords."""
    verified = df[df["has_verification"]]

    # Count predicted keywords
    pred_keywords = {}
    for keywords in df["predicted_keywords"]:
        for kw in keywords:
            pred_keywords[kw] = pred_keywords.get(kw, 0) + 1

    # Count actual keywords (verified only)
    actual_keywords = {}
    for keywords in verified["actual_keywords"]:
        for kw in keywords:
            actual_keywords[kw] = actual_keywords.get(kw, 0) + 1

    pred_df = pd.DataFrame([
        {"keyword": k, "count": v, "type": "predicted"}
        for k, v in sorted(pred_keywords.items(), key=lambda x: x[1], reverse=True)
    ])

    actual_df = pd.DataFrame([
        {"keyword": k, "count": v, "type": "actual"}
        for k, v in sorted(actual_keywords.items(), key=lambda x: x[1], reverse=True)
    ])

    return pred_df, actual_df


def build_field_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze field prediction patterns."""
    verified = df[df["has_verification"]]

    field_stats = []
    for _, row in verified.iterrows():
        for field in row["predicted_fields"]:
            match = 1 if field.lower() in [f.lower() for f in row["actual_fields"]] else 0
            field_stats.append({
                "field": field,
                "matched": match,
            })

    if not field_stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(field_stats)
    summary = stats_df.groupby("field").agg(
        total_predictions=("matched", "count"),
        matches=("matched", "sum"),
    ).reset_index()
    summary["accuracy"] = summary["matches"] / summary["total_predictions"]
    summary = summary.sort_values("total_predictions", ascending=False)

    return summary


def build_top_performers(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Get top performers by overall score."""
    verified = df[df["has_verification"]].copy()
    if verified.empty:
        return pd.DataFrame()

    top = verified.sort_values("overall_score", ascending=False).head(limit)
    return top[[
        "professor", "affiliation", "overall_score", "verdict",
        "keyword_precision", "keyword_recall", "keyword_f1",
        "field_precision", "field_recall", "field_f1",
        "n_papers"
    ]]


# -------------------- Dashboard --------------------
def create_dashboard(df: pd.DataFrame) -> Dash:
    """Create the Dash dashboard application."""

    # Get unique values for filters
    all_fields = sorted({f for fields in df["predicted_fields"] for f in fields})
    all_keywords = sorted({kw for keywords in df["predicted_keywords"] for kw in keywords})
    verdicts = ["all", "strong", "moderate", "weak"]

    app = Dash(__name__)
    app.title = "Future Work Prediction Explorer"

    app.layout = html.Div([
        html.H1("AI Professor Future Work Prediction Analysis"),
        html.P(
            "Interactive dashboard to explore LLM predictions of AI researchers' 2025 work, "
            "verified against actual 2025 publications from Semantic Scholar."
        ),

        # Filters
        html.Div([
            html.Div([
                html.Label("Verification Status"),
                dcc.RadioItems(
                    id="verification-filter",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Verified only", "value": "verified"},
                        {"label": "Unverified", "value": "unverified"},
                    ],
                    value="all",
                    inline=True,
                ),
            ], style=FILTER_CARD_STYLE),

            html.Div([
                html.Label("Score Range (verified only)"),
                dcc.RangeSlider(
                    id="score-filter",
                    min=0,
                    max=1,
                    step=0.05,
                    value=[0, 1],
                    marks={0: "0.0", 0.5: "0.5", 1.0: "1.0"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style=FILTER_CARD_STYLE),

            html.Div([
                html.Label("Verdict"),
                dcc.Dropdown(
                    id="verdict-filter",
                    options=[{"label": v.capitalize(), "value": v} for v in verdicts],
                    value="all",
                    clearable=False,
                ),
            ], style=FILTER_CARD_STYLE),

            html.Div([
                html.Label("Predicted Fields"),
                dcc.Dropdown(
                    id="field-filter",
                    options=[{"label": f, "value": f} for f in all_fields],
                    multi=True,
                    placeholder="Filter by field",
                ),
            ], style=FILTER_CARD_STYLE),

            html.Div([
                html.Label("Search Professor/Affiliation"),
                dcc.Input(
                    id="search-filter",
                    type="text",
                    placeholder="Type name or institution",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ], style=FILTER_CARD_STYLE),

            html.Div([
                html.Label("Download Filtered Data"),
                html.Button(
                    "Download CSV",
                    id="download-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "8px",
                        "backgroundColor": "#1b263b",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                ),
                dcc.Download(id="download-data"),
            ], style=FILTER_CARD_STYLE),
        ], style=FILTERS_GRID_STYLE),

        # Summary cards
        html.Div(id="summary-cards"),

        # Visualizations
        html.Div([
            dcc.Loading(dcc.Graph(id="score-distribution"), type="cube"),
            dcc.Loading(dcc.Graph(id="keyword-comparison"), type="cube"),
            dcc.Loading(dcc.Graph(id="field-accuracy"), type="cube"),
            dcc.Loading(dcc.Graph(id="precision-recall-scatter"), type="cube"),
        ], style=GRAPH_GRID_STYLE),

        # Tables
        html.Div([
            html.H3("Top Performers (by overall score)"),
            dash_table.DataTable(
                id="top-performers-table",
                columns=[
                    {"name": "Professor", "id": "professor"},
                    {"name": "Verdict", "id": "verdict"},
                    {"name": "Overall F1", "id": "overall_score"},
                    {"name": "KW F1", "id": "keyword_f1"},
                    {"name": "Field F1", "id": "field_f1"},
                    {"name": "Papers", "id": "n_papers"},
                ],
                data=[],
                sort_action="native",
                style_table={"maxHeight": "320px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "6px"},
            ),
        ], style={"margin": "24px 0"}),

        html.H2("Filtered Predictions"),
        dash_table.DataTable(
            id="predictions-table",
            columns=[
                {"name": "Professor", "id": "professor"},
                {"name": "Affiliation", "id": "affiliation"},
                {"name": "Verdict", "id": "verdict"},
                {"name": "Overall F1", "id": "overall_score"},
                {"name": "Keywords", "id": "predicted_keywords"},
                {"name": "Fields", "id": "predicted_fields"},
                {"name": "Papers", "id": "n_papers"},
                {"name": "Status", "id": "status"},
            ],
            data=[],
            page_current=0,
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "100px", "whiteSpace": "normal"},
            style_header={"backgroundColor": "#e0e6ef", "fontWeight": "600"},
        ),

        html.Footer([
            html.P(
                "Dashboard built with Plotly Dash. "
                "Export summary reports with --export flag."
            ),
        ], style={"marginTop": "40px"}),
    ], style=PAGE_STYLE)

    @app.callback(
        Output("score-distribution", "figure"),
        Output("keyword-comparison", "figure"),
        Output("field-accuracy", "figure"),
        Output("precision-recall-scatter", "figure"),
        Output("predictions-table", "data"),
        Output("summary-cards", "children"),
        Output("top-performers-table", "data"),
        Input("verification-filter", "value"),
        Input("score-filter", "value"),
        Input("field-filter", "value"),
        Input("verdict-filter", "value"),
        Input("search-filter", "value"),
    )
    def update_dashboard(verification_status, score_range, fields, verdict, search_query):
        min_score, max_score = score_range if score_range else (0, 1)

        filtered = filter_predictions(
            df,
            verification_status=None if verification_status == "all" else verification_status,
            min_score=min_score if verification_status != "unverified" else None,
            max_score=max_score if verification_status != "unverified" else None,
            fields=fields,
            verdict=verdict,
            search_query=search_query,
        )

        verified_count = filtered["has_verification"].sum()
        total_count = len(filtered)
        avg_score = filtered[filtered["has_verification"]]["overall_score"].mean() if verified_count > 0 else 0.0
        total_papers = filtered[filtered["has_verification"]]["n_papers"].sum()

        # Score distribution
        score_dist = build_score_distribution(filtered)
        if not score_dist.empty:
            score_fig = px.box(
                score_dist,
                x="metric",
                y="score",
                title="Precision, Recall & F1 Distribution by Metric Type",
                labels={"score": "Score", "metric": "Metric"},
            )
        else:
            score_fig = _empty_figure("Score Distribution", "No verified predictions")

        # Keyword comparison
        pred_kw, actual_kw = build_keyword_analysis(filtered)
        if not pred_kw.empty:
            # Check if we have actual keywords
            if not actual_kw.empty:
                combined_kw = pd.concat([pred_kw.head(20), actual_kw.head(20)])
                title = "Top 20 Predicted vs Actual Keywords"
            else:
                combined_kw = pred_kw.head(20)
                title = "Top 20 Predicted Keywords (actual keywords not available in data)"

            kw_fig = px.bar(
                combined_kw,
                x="count",
                y="keyword",
                color="type",
                orientation="h",
                title=title,
                barmode="group" if not actual_kw.empty else "relative",
            )
        else:
            kw_fig = _empty_figure("Keywords", "No keyword data")

        # Field accuracy
        field_analysis = build_field_analysis(filtered)
        if not field_analysis.empty:
            field_fig = px.bar(
                field_analysis.head(15),
                x="accuracy",
                y="field",
                orientation="h",
                title="Field Prediction Accuracy (Top 15)",
                labels={"accuracy": "Match Rate", "field": "Field"},
                text="total_predictions",
            )
            field_fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        else:
            field_fig = _empty_figure("Field Accuracy", "No field data")

        # Precision-Recall scatter
        verified_df = filtered[filtered["has_verification"]]
        if not verified_df.empty:
            pr_fig = px.scatter(
                verified_df,
                x="keyword_recall",
                y="keyword_precision",
                size="n_papers",
                color="verdict",
                hover_data=["professor", "overall_score"],
                title="Keyword Precision vs Recall (size = # papers, color = verdict)",
                labels={"keyword_recall": "Keyword Recall", "keyword_precision": "Keyword Precision"},
                color_discrete_map={"strong": "green", "moderate": "orange", "weak": "red"},
            )
        else:
            pr_fig = _empty_figure("Precision-Recall", "No verified predictions")

        # Table data
        table_data = filtered.copy()
        table_data["predicted_keywords"] = table_data["predicted_keywords"].apply(lambda x: ", ".join(x[:5]) + ("..." if len(x) > 5 else ""))
        table_data["predicted_fields"] = table_data["predicted_fields"].apply(lambda x: ", ".join(x))
        table_data["status"] = table_data["has_verification"].map({True: "Verified", False: "No papers"})
        table_data["overall_score"] = table_data["overall_score"].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
        table_data["verdict"] = table_data["verdict"].fillna("N/A")
        table_records = table_data[[
            "professor", "affiliation", "verdict", "overall_score",
            "predicted_keywords", "predicted_fields",
            "n_papers", "status"
        ]].to_dict("records")

        # Summary cards
        summary_cards = _render_summary_cards(total_count, verified_count, avg_score, total_papers)

        # Top performers
        top_perf = build_top_performers(filtered)
        if not top_perf.empty:
            top_perf["overall_score"] = top_perf["overall_score"].apply(lambda x: f"{x:.3f}")
            top_perf["keyword_f1"] = top_perf["keyword_f1"].apply(lambda x: f"{x:.3f}")
            top_perf["field_f1"] = top_perf["field_f1"].apply(lambda x: f"{x:.3f}")
        top_perf_data = top_perf.to_dict("records") if not top_perf.empty else []

        return (
            score_fig,
            kw_fig,
            field_fig,
            pr_fig,
            table_records,
            summary_cards,
            top_perf_data,
        )

    @app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"),
        State("verification-filter", "value"),
        State("score-filter", "value"),
        State("field-filter", "value"),
        State("verdict-filter", "value"),
        State("search-filter", "value"),
        prevent_initial_call=True,
    )
    def download_filtered(n_clicks, verification_status, score_range, fields, verdict, search_query):
        if not n_clicks:
            raise PreventUpdate

        min_score, max_score = score_range if score_range else (0, 1)

        filtered = filter_predictions(
            df,
            verification_status=None if verification_status == "all" else verification_status,
            min_score=min_score,
            max_score=max_score,
            fields=fields,
            verdict=verdict,
            search_query=search_query,
        )

        export_df = filtered.copy()
        export_df["predicted_keywords"] = export_df["predicted_keywords"].apply(lambda x: "; ".join(x))
        export_df["predicted_fields"] = export_df["predicted_fields"].apply(lambda x: "; ".join(x))
        export_df["actual_keywords"] = export_df["actual_keywords"].apply(lambda x: "; ".join(x))
        export_df["actual_fields"] = export_df["actual_fields"].apply(lambda x: "; ".join(x))

        csv_data = export_df[[
            "professor", "affiliation", "overall_score", "verdict",
            "keyword_precision", "keyword_recall", "keyword_f1",
            "field_precision", "field_recall", "field_f1",
            "n_papers", "predicted_keywords", "actual_keywords",
            "predicted_fields", "actual_fields",
        ]].to_csv(index=False)

        return dict(content=csv_data, filename="future_work_predictions_filtered.csv")

    return app


def _render_summary_cards(total: int, verified: int, avg_score: float, total_papers: int):
    """Render summary statistic cards."""
    text_style = {"margin": "4px 0 0", "fontSize": "14px", "opacity": 0.85}
    value_style = {"margin": "0", "fontSize": "28px"}

    return html.Div([
        html.Div([
            html.H3(f"{total}", style=value_style),
            html.P("Total Professors", style=text_style),
        ], style=SUMMARY_CARD_STYLE),
        html.Div([
            html.H3(f"{verified}", style=value_style),
            html.P("With Verification", style=text_style),
        ], style=SUMMARY_CARD_STYLE),
        html.Div([
            html.H3(f"{avg_score:.3f}", style=value_style),
            html.P("Avg Overall Score", style=text_style),
        ], style=SUMMARY_CARD_STYLE),
        html.Div([
            html.H3(f"{total_papers}", style=value_style),
            html.P("Total 2025 Papers", style=text_style),
        ], style=SUMMARY_CARD_STYLE),
    ], style=SUMMARY_GRID_STYLE)


def _empty_figure(title: str, subtitle: str) -> dict:
    """Create an empty figure with a message."""
    fig = px.scatter(x=[], y=[])
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                text=subtitle,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
    return fig


# -------------------- Export Reports --------------------
def export_reports(df: pd.DataFrame) -> None:
    """Export summary CSV reports."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    verified = df[df["has_verification"]]

    # Overall summary
    summary = verified[[
        "professor", "affiliation", "overall_score", "verdict",
        "keyword_precision", "keyword_recall", "keyword_f1",
        "field_precision", "field_recall", "field_f1",
        "n_papers"
    ]].copy()
    summary.to_csv(SUMMARY_CSV, index=False)

    # Keyword analysis
    pred_kw, actual_kw = build_keyword_analysis(df)
    combined_kw = pd.concat([pred_kw, actual_kw])
    combined_kw.to_csv(KEYWORD_CSV, index=False)

    # Field analysis
    field_analysis = build_field_analysis(df)
    if not field_analysis.empty:
        field_analysis.to_csv(FIELD_CSV, index=False)

    # Top performers
    top_perf = build_top_performers(df, limit=50)
    if not top_perf.empty:
        top_perf.to_csv(TOP_PERFORMERS_CSV, index=False)

    print(f"\nExported summary reports:")
    print(f"  - Overall summary → {SUMMARY_CSV}")
    print(f"  - Keyword analysis → {KEYWORD_CSV}")
    print(f"  - Field analysis → {FIELD_CSV}")
    print(f"  - Top performers → {TOP_PERFORMERS_CSV}")
    print(f"\nVerified predictions: {len(verified)}/{len(df)}")
    print(f"Average overall score: {verified['overall_score'].mean():.3f}")

    # Verdict breakdown
    if not verified.empty:
        verdict_counts = verified["verdict"].value_counts()
        print(f"\nVerdict breakdown:")
        for verdict, count in verdict_counts.items():
            print(f"  - {verdict}: {count}")


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the future work prediction dashboard or export summary tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              Launch the dashboard (default on http://127.0.0.1:8050)
                python analysis/future_work_dashboard.py

              Export CSV summaries only
                python analysis/future_work_dashboard.py --export
        """),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--export", action="store_true", help="Export CSVs and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_predictions()

    if args.export:
        export_reports(df)
        return

    app = create_dashboard(df)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
