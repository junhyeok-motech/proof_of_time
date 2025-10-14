#!/usr/bin/env python3

"""Generate the Inspect MCQ dataset from sandbox and award spreadsheets."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


DEFAULT_CHOICES = ["Findings", "Main", "Outstanding", "Best"]
TARGET_SHEETS = {"ACL", "EMNLP", "NAACL"}


def normalize_key(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())


def clean_abstract(raw: str | None) -> str:
    if not isinstance(raw, str):
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = ast.literal_eval(stripped)
        except Exception:
            data = None
        if isinstance(data, dict):
            if "#text" in data and isinstance(data["#text"], str):
                return data["#text"].strip()
            parts = [str(v).strip() for v in data.values() if isinstance(v, str)]
            if parts:
                return " ".join(parts)
    return stripped


def parse_year(conference: str | None) -> int | None:
    if not isinstance(conference, str):
        return None
    match = re.search(r"(19|20)\d{2}", conference)
    if match:
        return int(match.group(0))
    return None


def classify_category(category: str) -> str:
    lowered = category.lower()
    if "outstanding" in lowered:
        return "Outstanding"
    if "best" in lowered or "runner-up" in lowered or "runner up" in lowered:
        return "Best"
    if "findings" in lowered:
        return "Findings"
    return "Main"


def iter_award_rows(workbook: Path, sheets: Iterable[str]) -> Iterable[Dict]:
    xl = pd.ExcelFile(workbook)
    for sheet in xl.sheet_names:
        if sheet not in sheets:
            continue
        data = xl.parse(sheet)
        for row in data.itertuples(index=False):
            yield sheet, row._asdict()


def build_lookup(accepted_csv: Path) -> Dict[str, Dict]:
    df = pd.read_csv(accepted_csv)
    lookup: Dict[str, Dict] = {}
    for row in df.itertuples(index=False):
        key = normalize_key(row.title)
        if key:
            lookup.setdefault(key, row._asdict())
    return lookup


def generate_records(
    accepted_csv: Path,
    awards_xlsx: Path,
    *,
    random_seed: int | None = None,
) -> tuple[list[Dict], list[Dict]]:
    rng = random.Random(random_seed)
    lookup = build_lookup(accepted_csv)
    records: list[Dict] = []
    skipped: list[Dict] = []

    for sheet, award_row in iter_award_rows(awards_xlsx, TARGET_SHEETS):
        key = normalize_key(award_row.get("title"))
        if not key or key not in lookup:
            skipped.append({"sheet": sheet, "title": award_row.get("title"), "reason": "missing in accepted CSV"})
            continue

        paper = lookup[key]
        abstract = clean_abstract(paper.get("abstract"))
        if not abstract:
            skipped.append({"sheet": sheet, "title": award_row.get("title"), "reason": "missing abstract"})
            continue

        category = str(award_row.get("category", "") or "")
        answer = classify_category(category)

        include_authors = False
        authors_line = None
        authors = paper.get("authors")
        if isinstance(authors, str) and authors.strip():
            include_authors = rng.random() < 0.5
            if include_authors:
                authors_line = f"Authors: {authors.strip()}"

        context_lines = [f"Title: {str(award_row.get('title')).strip()}"]
        context_lines.append(f"Abstract: {abstract}")
        if authors_line:
            context_lines.append(authors_line)

        conference = str(award_row.get("conference", "") or "")
        year = parse_year(conference)
        cutoff_period = "future_2025" if year and year >= 2025 else "pre_2025"

        metadata = {
            "conference": conference,
            "category": category,
            "sheet": sheet,
            "accepted_tags": str(paper.get("tags", "")),
            "authors_included": include_authors,
            "year": year,
            "cutoff_period": cutoff_period,
        }

        record = {
            "question": "Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?",
            "choices": DEFAULT_CHOICES,
            "answer": answer,
            "context": "\n".join(context_lines),
            "metadata": metadata,
        }
        records.append(record)

    return records, skipped


def write_jsonl(records: Iterable[Dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MCQ dataset used by the Inspect benchmark.")
    parser.add_argument(
        "--accepted-csv",
        type=Path,
        default=Path("inspect/emnlp_react/sandbox/data/accepted_papers.csv"),
        help="Path to accepted papers CSV exported for the sandbox.",
    )
    parser.add_argument(
        "--awards-xlsx",
        type=Path,
        default=Path("data/pot-best-papers.xlsx"),
        help="Excel workbook containing award categories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inspect/emnlp_react/mcq_dataset.jsonl"),
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Seed for author inclusion ablation randomness.",
    )

    args = parser.parse_args()

    records, skipped = generate_records(
        accepted_csv=args.accepted_csv,
        awards_xlsx=args.awards_xlsx,
        random_seed=args.seed,
    )
    write_jsonl(records, args.output)

    print(f"[write] {len(records)} records -> {args.output}")
    if skipped:
        print(f"[skip] {len(skipped)} rows skipped (examples shown below)")
        for sample in skipped[:10]:
            print(f"  - {sample['sheet']}: {sample['title']} ({sample['reason']})")


if __name__ == "__main__":
    main()

