#!/usr/bin/env python3

"""Sample historical main and findings papers for the MCQ benchmark."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasets import load_dataset


DEFAULT_CHOICES = ["Findings", "Main", "Outstanding", "Best"]
TARGET_PREFIXES = {"acl", "emnlp", "naacl"}
TARGET_TRACKS = {"main", "findings"}


def normalize_key(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.lower().split())


def load_awarded_titles(xlsx_path: Path) -> set[str]:
    xl = pd.ExcelFile(xlsx_path)
    titles: set[str] = set()
    for sheet in xl.sheet_names:
        if sheet not in {"ACL", "EMNLP", "NAACL"}:
            continue
        df = xl.parse(sheet)
        for title in df["title"].dropna():
            key = normalize_key(title)
            if key:
                titles.add(key)
    return titles


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


def parse_year(source_split: str, tags: str | None) -> int | None:
    # Source split format: "<conf>_<track>_<year>"
    parts = source_split.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    if isinstance(tags, str):
        match = re.search(r"(19|20)\d{2}", tags)
        if match:
            return int(match.group(0))
    return None


@dataclass
class Candidate:
    title: str
    abstract: str
    authors: str
    conference_tag: str
    source_dataset: str
    source_split: str
    track: str
    year: int | None


def gather_candidates(awarded_titles: set[str]) -> Dict[Tuple[str, str], List[Candidate]]:
    dataset = load_dataset("shanchen/Accepted-Papers-Aggregated", split="train")
    buckets: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
    seen_titles_by_combo: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for record in dataset:
        source_split = (record.get("source_split") or "").lower()
        if not source_split:
            continue
        parts = source_split.split("_")
        if len(parts) < 2:
            continue
        prefix, track = parts[0], parts[1]
        if prefix not in TARGET_PREFIXES or track not in TARGET_TRACKS:
            continue

        title = record.get("title")
        norm_title = normalize_key(title)
        if not title or not norm_title:
            continue
        if norm_title in awarded_titles:
            continue
        combo = (prefix, track)
        if norm_title in seen_titles_by_combo[combo]:
            continue

        abstract = clean_abstract(record.get("abstract") or record.get("abs"))
        if not abstract:
            continue

        candidate = Candidate(
            title=title.strip(),
            abstract=abstract,
            authors=(record.get("authors") or "").strip(),
            conference_tag=str(record.get("tags") or ""),
            source_dataset=str(record.get("source_dataset") or ""),
            source_split=record.get("source_split") or "",
            track=track,
            year=parse_year(record.get("source_split") or "", record.get("tags")),
        )
        buckets[combo].append(candidate)
        seen_titles_by_combo[combo].add(norm_title)

    return buckets


def sample_candidates(
    buckets: Dict[Tuple[str, str], List[Candidate]],
    count_per_track: int,
    rng: random.Random,
) -> List[Tuple[str, str, Candidate]]:
    sampled: List[Tuple[str, str, Candidate]] = []
    for key, candidates in buckets.items():
        prefix, track = key
        if not candidates:
            continue
        rng.shuffle(candidates)
        take = candidates if count_per_track <= 0 else candidates[:count_per_track]
        sampled.extend((prefix, track, cand) for cand in take)
    return sampled


def build_record(prefix: str, track: str, candidate: Candidate, rng: random.Random) -> Dict:
    include_authors = bool(candidate.authors) and rng.random() < 0.5
    context_lines = [
        f"Title: {candidate.title}",
        f"Abstract: {candidate.abstract}",
    ]
    if include_authors:
        context_lines.append(f"Authors: {candidate.authors}")

    conference = candidate.conference_tag or f"{prefix.upper()} {candidate.year or ''}".strip()
    cutoff_period = "future_2025" if candidate.year and candidate.year >= 2025 else "pre_2025"

    metadata = {
        "conference": conference,
        "track": track,
        "conference_group": prefix,
        "source_split": candidate.source_split,
        "source_dataset": candidate.source_dataset,
        "authors_included": include_authors,
        "year": candidate.year,
        "cutoff_period": cutoff_period,
    }

    return {
        "question": "Which recognition tier (Findings/Main/Outstanding/Best) best fits this paper?",
        "choices": DEFAULT_CHOICES,
        "answer": "Findings" if track == "findings" else "Main",
        "context": "\n".join(context_lines),
        "metadata": metadata,
    }


def write_jsonl(records: Iterable[Dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate historical main/findings MCQ samples from the accepted papers dataset."
    )
    parser.add_argument(
        "--awards-xlsx",
        type=Path,
        default=Path("data/pot-best-papers.xlsx"),
        help="Workbook containing award-winning papers to exclude.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inspect/emnlp_react/historical_mcq_dataset.jsonl"),
        help="Destination JSONL file for sampled MCQ records.",
    )
    parser.add_argument(
        "--count-per-track",
        type=int,
        default=25,
        help="Number of samples to draw for each (conference, track) combination. Use 0 to include all available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    awarded_titles = load_awarded_titles(args.awards_xlsx)
    buckets = gather_candidates(awarded_titles)

    sampled = sample_candidates(buckets, args.count_per_track, rng)
    records = [build_record(prefix, track, candidate, rng) for prefix, track, candidate in sampled]

    write_jsonl(records, args.output)

    # Print summary
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for prefix, track, _ in sampled:
        summary[prefix][track] += 1

    print(f"[write] {len(records)} records -> {args.output}")
    for prefix in sorted(summary):
        for track in sorted(summary[prefix]):
            print(f"  - {prefix.upper()} {track}: {summary[prefix][track]} samples")
    missing = [
        (pref, track)
        for pref in TARGET_PREFIXES
        for track in TARGET_TRACKS
        if (pref, track) not in buckets or not buckets[(pref, track)]
    ]
    if missing:
        print("[warn] No candidates available for:", ", ".join(f"{p.upper()} {t}" for p, t in missing))


if __name__ == "__main__":
    main()
