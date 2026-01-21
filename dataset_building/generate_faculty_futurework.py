#!/usr/bin/env python3

"""Generate Inspect datasets + sandbox assets for the faculty future-work benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

FIELD_KEYWORDS: Dict[str, Sequence[str]] = {
    "AI Safety & Alignment": [
        "alignment",
        "aligned",
        "safety",
        "robust",
        "robustness",
        "adversarial",
        "certified",
        "verification",
        "interpretability",
        "explanation",
        "fairness",
        "bias",
        "privacy",
        "secure",
    ],
    "Foundation Models & LLMs": [
        "language model",
        "large language",
        "llm",
        "transformer",
        "pretrain",
        "instruction",
        "few-shot",
        "prompt",
        "in-context",
        "decoder",
        "token",
        "retrieval",
    ],
    "Robotics & Embodied AI": [
        "robot",
        "manipulation",
        "control",
        "dynamics",
        "motion",
        "locomotion",
        "navigation",
        "policy",
        "embodied",
        "autonomous vehicle",
    ],
    "Vision & Multimodal Learning": [
        "image",
        "vision",
        "video",
        "multimodal",
        "perception",
        "object detection",
        "segmentation",
        "diffusion",
        "render",
        "3d",
    ],
    "Economics, Policy & Society": [
        "economics",
        "policy",
        "market",
        "platform",
        "society",
        "labor",
        "regulation",
        "law",
        "education",
        "impact",
    ],
    "Healthcare & Biomedicine": [
        "health",
        "medical",
        "patient",
        "clinical",
        "biomedical",
        "protein",
        "drug",
        "genomic",
        "hospital",
        "diagnosis",
    ],
    "Climate & Sustainability": [
        "climate",
        "carbon",
        "emission",
        "energy",
        "sustainable",
        "weather",
        "environment",
        "greenhouse",
        "renewable",
        "wildfire",
    ],
    "Optimization, Theory & ML Systems": [
        "optimization",
        "theory",
        "convergence",
        "generalization",
        "complexity",
        "gradient",
        "proof",
        "analysis",
        "compiler",
        "systems",
    ],
}
FIELD_CHOICES = list(FIELD_KEYWORDS.keys()) + ["General AI Research"]


@dataclass
class Publication:
    professor: str
    scholar_id: str
    title: str
    authors: str
    abstract: str
    venue: str
    year: int
    field: str
    matched_keywords: List[str]
    source_file: str


def slug_to_name_and_id(stem: str) -> Tuple[str, str]:
    """Split a filename stem into a human-readable professor name + scholar id."""
    if "_" not in stem:
        name = stem.replace("_", " ").strip()
        return " ".join(name.split()), ""
    name_part, scholar_id = stem.rsplit("_", 1)
    tokens = name_part.split("_")
    while tokens and re.fullmatch(r"[A-Z]{4,}", tokens[-1]):
        # Handle cases where the scholar id leaked into the name (e.g., ZCniP_MAAAAJ)
        suffix = tokens.pop()
        scholar_id = f"{suffix}_{scholar_id}" if scholar_id else suffix
    name = " ".join(tokens).replace("-", " ")
    return " ".join(name.split()), scholar_id


def normalize_text(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value.strip())


def infer_field(title: str, abstract: str, venue: str) -> Tuple[str, List[str]]:
    """Lightweight keyword matcher returning (field, hits)."""
    haystack = " ".join(part for part in [title, abstract, venue] if part).lower()
    best_field = "General AI Research"
    best_score = 0
    best_hits: List[str] = []
    for field, keywords in FIELD_KEYWORDS.items():
        hits: List[str] = []
        score = 0
        for keyword in keywords:
            if keyword in haystack:
                score += 1
                hits.append(keyword)
        if score > best_score:
            best_field = field
            best_score = score
            best_hits = hits
    return best_field, best_hits


def iter_publications(source_dir: Path) -> Iterable[Publication]:
    for csv_path in sorted(source_dir.glob("*.csv")):
        professor, scholar_id = slug_to_name_and_id(csv_path.stem)
        with csv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    year = int(row.get("Year", "").strip())
                except Exception:
                    continue
                title = normalize_text(row.get("Title"))
                abstract = normalize_text(row.get("Abstract"))
                venue = normalize_text(row.get("Venue"))
                authors = normalize_text(row.get("Authors"))
                field, hits = infer_field(title, abstract, venue)
                yield Publication(
                    professor=professor,
                    scholar_id=scholar_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    venue=venue,
                    year=year,
                    field=field,
                    matched_keywords=hits,
                    source_file=csv_path.name,
                )


def write_jsonl(records: Iterable[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_professor_field_records(
    by_professor: Dict[str, List[Publication]], rng: random.Random
) -> List[dict]:
    records: List[dict] = []
    for professor, pubs in by_professor.items():
        field_counts = Counter(pub.field for pub in pubs)
        if not field_counts:
            continue
        correct_field, _ = field_counts.most_common(1)[0]
        titles = sorted({pub.title for pub in pubs})
        sample_titles = titles[:4]
        distractors = [label for label in FIELD_CHOICES if label != correct_field]
        if len(distractors) < 3:
            continue
        options = rng.sample(distractors, 3) + [correct_field]
        rng.shuffle(options)
        records.append(
            {
                "question": (
                    f"Professor {professor} has the following 2025 publications. "
                    "Which field best captures their 2025 agenda?"
                ),
                "prompt": (
                    f"Professor: {professor}\n"
                    "2025 Titles:\n- " + "\n- ".join(sample_titles)
                ),
                "choices": options,
                "answer": correct_field,
                "metadata": {
                    "professor": professor,
                    "num_titles": len(titles),
                },
            }
        )
    return records


def build_professor_article_records(
    by_professor: Dict[str, List[Publication]],
    distractor_pool: List[Publication],
    professors_without_2025: List[str],
    rng: random.Random,
) -> List[dict]:
    def format_option(pub: Publication) -> str:
        return f"Title: {pub.title}\nAbstract: {pub.abstract or 'N/A'}\nVenue: {pub.venue} ({pub.year})"

    records: List[dict] = []
    letters = ["A", "B", "C", "D"]

    pool_by_prof = {prof: pubs for prof, pubs in by_professor.items() if pubs}
    for professor, pubs in pool_by_prof.items():
        correct_pub = rng.choice(pubs)
        wrong = [p for p in distractor_pool if p.professor != professor]
        if len(wrong) < 3:
            continue
        distractors = rng.sample(wrong, 3)
        options = distractors + [correct_pub]
        rng.shuffle(options)
        letter_map = {letter: opt for letter, opt in zip(letters, options)}
        correct_letter = next(letter for letter, pub in letter_map.items() if pub == correct_pub)
        context_lines = [
            f"Professor: {professor}",
            "Which of the following 2025 papers did they author/co-author?",
        ]
        for letter in letters:
            entry = letter_map[letter]
            context_lines.append(f"{letter}. {format_option(entry)}")
        context_lines.append("None. None of these papers belong to the professor.")
        records.append(
            {
                "question": (
                    "Respond with A, B, C, D, or None to indicate which paper belongs to the professor."
                ),
                "prompt": "\n".join(context_lines),
                "choices": ["A", "B", "C", "D", "None"],
                "answer": correct_letter,
                "metadata": {
                    "professor": professor,
                    "correct_title": correct_pub.title,
                },
            }
        )

    # Add some "None" cases for professors without 2025 publications
    none_professors = rng.sample(professors_without_2025, min(len(professors_without_2025), 25))
    for professor in none_professors:
        if len(distractor_pool) < 4:
            break
        options = rng.sample(distractor_pool, 4)
        context_lines = [
            f"Professor: {professor}",
            "Which of the following 2025 papers did they author/co-author?",
        ]
        for letter, pub in zip(letters, options):
            context_lines.append(f"{letter}. {format_option(pub)}")
        context_lines.append("None. None of these papers belong to the professor.")
        records.append(
            {
                "question": "Respond with A, B, C, D, or None.",
                "prompt": "\n".join(context_lines),
                "choices": ["A", "B", "C", "D", "None"],
                "answer": "None",
                "metadata": {
                    "professor": professor,
                    "correct_title": None,
                },
            }
        )
    return records


def build_field_focus_records(
    field_to_pubs: Dict[str, List[Publication]],
    rng: random.Random,
) -> List[dict]:
    records: List[dict] = []
    for field, pubs in field_to_pubs.items():
        if len(pubs) < 4:
            continue
        sample_count = min(4, len(pubs))
        sampled = rng.sample(pubs, sample_count)
        distractors = [label for label in FIELD_CHOICES if label != field]
        if len(distractors) < 3:
            continue
        options = rng.sample(distractors, 3) + [field]
        rng.shuffle(options)
        context = ["Representative 2025 papers:"]
        for idx, pub in enumerate(sampled, start=1):
            context.append(
                f"{idx}. {pub.title} — {pub.abstract[:280]}{'...' if len(pub.abstract) > 280 else ''}"
            )
        records.append(
            {
                "question": "Which field best matches the collection of papers?",
                "prompt": "\n".join(context),
                "choices": options,
                "answer": field,
                "metadata": {
                    "field": field,
                    "num_papers": len(pubs),
                },
            }
        )
    return records


def copy_raw_csvs(source_dir: Path, sandbox_dir: Path) -> None:
    target_dir = sandbox_dir / "faculty_publications"
    target_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in source_dir.glob("*.csv"):
        destination = target_dir / csv_path.name
        shutil.copy2(csv_path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets for the faculty future-work Inspect benchmark.")
    parser.add_argument("--source-dir", type=Path, default=Path("faculty_publications"))
    parser.add_argument(
        "--sandbox-dir",
        type=Path,
        default=Path("proof_of_time/data"),
        help="Directory to store sandbox artifacts (JSONL + raw CSV copies).",
    )
    parser.add_argument(
        "--professor-field-output",
        type=Path,
        default=Path("inspect/future_work_react/professor_field_mcq.jsonl"),
    )
    parser.add_argument(
        "--professor-article-output",
        type=Path,
        default=Path("inspect/future_work_react/professor_article_mcq.jsonl"),
    )
    parser.add_argument(
        "--field-focus-output",
        type=Path,
        default=Path("inspect/future_work_react/field_focus_mcq.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    source_dir = args.source_dir
    sandbox_dir = args.sandbox_dir
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    publications = list(iter_publications(source_dir))
    if not publications:
        raise SystemExit(f"No publications found under {source_dir}")

    copy_raw_csvs(source_dir, sandbox_dir)

    aggregated_path = sandbox_dir / "faculty_publications.jsonl"
    write_jsonl(
        (
            {
                "professor": pub.professor,
                "scholar_id": pub.scholar_id,
                "title": pub.title,
                "authors": pub.authors,
                "abstract": pub.abstract,
                "venue": pub.venue,
                "year": pub.year,
                "field": pub.field,
                "matched_keywords": pub.matched_keywords,
                "source_file": pub.source_file,
            }
            for pub in publications
        ),
        aggregated_path,
    )

    publications_2025 = [pub for pub in publications if pub.year == 2025]
    by_professor = defaultdict(list)
    field_to_pubs = defaultdict(list)
    professors_all: set[str] = set()
    for pub in publications_2025:
        by_professor[pub.professor].append(pub)
        field_to_pubs[pub.field].append(pub)
    for pub in publications:
        professors_all.add(pub.professor)
    professors_without_2025 = sorted(professors_all - set(by_professor.keys()))

    professor_field_records = build_professor_field_records(by_professor, rng)
    professor_article_records = build_professor_article_records(
        by_professor, publications_2025, professors_without_2025, rng
    )
    field_focus_records = build_field_focus_records(field_to_pubs, rng)

    write_jsonl(professor_field_records, args.professor_field_output)
    write_jsonl(professor_article_records, args.professor_article_output)
    write_jsonl(field_focus_records, args.field_focus_output)

    print(f"[write] {len(publications)} rows -> {aggregated_path}")
    print(f"[dataset] professor_field_mcq: {len(professor_field_records)} records")
    print(f"[dataset] professor_article_mcq: {len(professor_article_records)} records")
    print(f"[dataset] field_focus_mcq: {len(field_focus_records)} records")


if __name__ == "__main__":
    main()
