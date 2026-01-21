"""Inspect AI benchmark for citation prediction tasks.

This benchmark evaluates LLMs' ability to predict citation counts for awarded
NLP papers. The agent operates inside a sandbox with:
- Historical EMNLP papers (2021-2024) with citation metadata
- Best paper data with actual citation counts

Tasks:
1. Multiple choice: Which of 4 papers has highest citations?
2. Ranking: Rank 4 papers by citation count
3. Bucket prediction: Predict citation range (5 buckets)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import answer, match
from inspect_ai.solver import generate, system_message
from inspect_ai.tool import bash, bash_session, python, text_editor, think

# Add parent directory to path to import common module
_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))

from common.prompt_utils import get_offline_preamble
from common.task_config import (
    AGENT_TASK_CONFIG,
    AGENT_TASK_LIMITS,
    SIMPLE_TASK_CONFIG,
    SIMPLE_TASK_LIMITS,
)

SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
MCQ_PATH = Path(__file__).resolve().parent / "multiple_choice.jsonl"
RANKING_PATH = Path(__file__).resolve().parent / "ranking.jsonl"
BUCKET_PATH = Path(__file__).resolve().parent / "bucket_prediction.jsonl"


def _load_samples(path: Path) -> Iterable[Sample]:
    """Parse a local JSONL dataset into Inspect `Sample` objects."""
    sandbox_mount = str(SANDBOX_ROOT.resolve())
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            payload = json.loads(line)
            question = payload.get("question", "")
            input_text = question
            context = payload.get("context")
            choices = payload.get("choices")
            if context:
                input_parts = [context.strip(), f"Question: {question}"]
                if choices:
                    options = ", ".join(choices)
                    input_parts.append(f"Options: {options}")
                input_text = "\n\n".join(part for part in input_parts if part)

            yield Sample(
                id=idx,
                input=input_text,
                target=payload["answer"],
                metadata=payload.get("metadata", {}),
                files={"": sandbox_mount},
            )


def build_mcq_dataset() -> Dataset:
    """Create an Inspect dataset from the multiple choice JSONL file."""
    samples: List[Sample] = list(_load_samples(MCQ_PATH))
    return MemoryDataset(samples)


def build_ranking_dataset() -> Dataset:
    """Create an Inspect dataset from the ranking JSONL file."""
    samples: List[Sample] = list(_load_samples(RANKING_PATH))
    return MemoryDataset(samples)


def build_bucket_prediction_dataset() -> Dataset:
    """Create an Inspect dataset from the bucket prediction JSONL file."""
    samples: List[Sample] = list(_load_samples(BUCKET_PATH))
    return MemoryDataset(samples)


def build_multiple_choice_agent(use_offline_prompt: bool = True):
    """Configure a React agent for citation prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="citation-mc-agent",
        prompt=(
            f"{offline_prefix}"
            "You are a research impact analyst working inside a sandbox with "
            "historical citation data. "
            "Your task: Given a set of four papers indexed as {A, B, C, D}"
            "Predict which paper has the highest number of citation based on historical patterns.\n\n"
            "Strategy:\n"
            "1. Load and explore the historical database (2021-2024 papers)\n"
            "2. Identify patterns: Which topics get high citations?\n"
            "3. For each paper in the paper set, search for similar papers in history\n"
            "4. Pick the paper from the set that should have the highest citation numbers\n"
            "5. Respond with ONLY ONE LETTER: A, B, C, or D\n\n"
            "CRITICAL: After using tools to investigate, provide your final answer as a SINGLE LETTER on its own line. "
            "Do not add explanations after your answer. The answer must be exactly one of: A, B, C, D"
        ),
        tools=[
            think(),
            python(),
            bash(),
            bash_session(),
            text_editor(),
        ],
    )


def build_ranking_agent(use_offline_prompt: bool = True):
    """Configure a React agent for citation prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="citation-ranking-agent",
        prompt=(
            f"{offline_prefix}"
            "You are a research impact analyst working inside a sandbox with "
            "historical citation data. Your goal is to rank papers by their citation counts "
            "using patterns from 2021-2024 data.\n\n"
            "Available data in sandbox:\n"
            "- historical_papers_2021_2024.jsonl: 38,330 papers with citation counts, "
            "authors, titles, years, and venues\n\n"
            "Your task: Rank four papers (labeled A, B, C, D) from highest to lowest citations "
            "based on historical patterns.\n\n"
            "Strategy:\n"
            "1. Load and explore the historical database (2021-2024 papers)\n"
            "2. Identify citation patterns: Which topics, venues, or paper characteristics "
            "correlate with high citations?\n"
            "3. For each paper in the set {A, B, C, D}, search for similar papers in the "
            "historical data\n"
            "4. Compare expected citation counts based on historical similarities\n"
            "5. Rank the papers from highest to lowest predicted citations\n"
            "6. Respond with ONLY the ranking as four letters separated by commas\n\n"
            "CRITICAL: Provide your final answer as a comma-separated sequence on its own line.\n"
            "Format: A, B, C, D (where the first letter is highest citations, last is lowest)\n"
            "Example answers: 'B, A, D, C' or 'C, D, A, B'\n"
            "Do not add explanations after your answer."
        ),
        tools=[
            think(),
            python(),
            bash(),
            bash_session(),
            text_editor(),
        ],
    )


def build_bucket_prediction_agent(use_offline_prompt: bool = True):
    """Configure a React agent for citation prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="citation-bucket-prediction-agent",
        prompt=(
            f"{offline_prefix}"
            "You are a research impact analyst working inside a sandbox with "
            "historical citation data. Your goal is to predict citation ranges "
            "for papers using patterns from 2021-2024 data.\n\n"
            "Available data in sandbox:\n"
            "- historical_papers_2021_2024.jsonl: 38,330 papers with citation counts, "
            "authors, titles, years, and venues\n\n"
            "Your task: Given a paper, predict which citation range bucket it falls into.\n\n"
            "Citation buckets:\n"
            "- A: 0-10 citations (very low impact)\n"
            "- B: 10-25 citations (low impact)\n"
            "- C: 25-60 citations (moderate impact)\n"
            "- D: 60-150 citations (high impact)\n"
            "- E: 150+ citations (very high impact)\n\n"
            "Strategy:\n"
            "1. Load and explore the historical database (2021-2024 papers)\n"
            "2. Analyze citation distributions: What percentage of papers fall into each bucket?\n"
            "3. Identify patterns: Which paper characteristics (topics, venues, authors) "
            "correlate with each citation range?\n"
            "4. Search for similar papers in the historical data\n"
            "5. Based on similar papers' citation counts, determine the most likely bucket\n"
            "6. Respond with ONLY ONE LETTER: A, B, C, D, or E\n\n"
            "CRITICAL: Provide your final answer as a single letter on its own line.\n"
            "Do not add explanations after your answer. The answer must be exactly one of: A, B, C, D, E"
        ),
        tools=[
            think(),
            python(),
            bash(),
            bash_session(),
            text_editor(),
        ],
    )


@task()
def citation_multiple_choice() -> Task:
    """Multiple choice: Which of 4 papers has the highest citations?"""
    dataset = build_mcq_dataset()
    agent = build_multiple_choice_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_mc"},
    )

@task()
def citation_multiple_choice_simple_task() -> Task:
    """Multiple choice: Which of 4 papers has the highest citations? zeroshot prediction"""
    dataset = build_mcq_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are a research impact analyst. Your goal is to predict which paper "
                "will have the highest citation count based on the paper titles and abstracts provided.\n\n"
                "Consider factors like:\n"
                "- Novelty and significance of the research problem\n"
                "- Broad applicability vs. niche focus\n"
                "- Clarity and appeal of the core contribution\n"
                "- Potential impact on the field\n\n"
                "Respond with ONLY ONE LETTER: A, B, C, or D\n"
                "Do not explain your reasoning. Just output the single letter corresponding to the paper "
                "you predict will have the highest citations."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_mc_simple"},
    )


@task()
def citation_multiple_choice_local() -> Task:
    """Multiple choice without Docker sandbox (direct file access)."""
    dataset = build_mcq_dataset()
    agent = build_multiple_choice_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_mc_local"},
    )


@task()
def citation_multiple_choice_no_offline_prompt() -> Task:
    """Multiple choice without the shared offline Antigravity preamble."""
    dataset = build_mcq_dataset()
    agent = build_multiple_choice_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_mc_no_offline"},
    )


@task()
def citation_multiple_choice_no_offline_prompt_local() -> Task:
    """Multiple choice (no preamble) without Docker sandbox."""
    dataset = build_mcq_dataset()
    agent = build_multiple_choice_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_mc_no_offline_local"},
    )


@task()
def citation_ranking() -> Task:
    """Ranking: Rank 4 papers from most to least cited."""
    dataset = build_ranking_dataset()
    agent = build_ranking_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_ranking"},
    )


@task()
def citation_ranking_local() -> Task:
    """Ranking task without Docker sandbox (direct file access)."""
    dataset = build_ranking_dataset()
    agent = build_ranking_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_ranking_local"},
    )


@task()
def citation_ranking_no_offline_prompt() -> Task:
    """Ranking task without the shared offline Antigravity preamble."""
    dataset = build_ranking_dataset()
    agent = build_ranking_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_ranking_no_offline"},
    )


@task()
def citation_ranking_no_offline_prompt_local() -> Task:
    """Ranking task (no preamble) without Docker sandbox."""
    dataset = build_ranking_dataset()
    agent = build_ranking_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_ranking_no_offline_local"},
    )


@task()
def citation_ranking_simple_task() -> Task:
    """Ranking: Rank 4 papers from most to least cited. zeroshot prediction"""
    dataset = build_ranking_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are a research impact analyst. Your goal is to rank four papers "
                "by their predicted citation counts based on the titles and abstracts provided.\n\n"
                "Consider factors like:\n"
                "- Novelty and significance of the research problem\n"
                "- Broad applicability vs. niche focus\n"
                "- Clarity and appeal of the core contribution\n"
                "- Potential impact on the field\n"
                "- Relevance to current research trends\n\n"
                "Your task: Rank the four papers (labeled A, B, C, D) from highest to lowest "
                "predicted citations.\n\n"
                "CRITICAL: Provide your final answer as a comma-separated sequence on its own line.\n"
                "Format: A, B, C, D (where the first letter is highest citations, last is lowest)\n"
                "Example answers: 'B, A, D, C' or 'C, D, A, B'\n"
                "Do not explain your reasoning. Just output the ranking."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_ranking_simple"},
    )


@task()
def citation_bucket_prediction() -> Task:
    """Bucket prediction: Predict citation range (5 buckets: 0-10, 10-25, 25-60, 60-150, 150+)."""
    dataset = build_bucket_prediction_dataset()
    agent = build_bucket_prediction_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_bucket"},
    )


@task()
def citation_bucket_prediction_local() -> Task:
    """Bucket prediction without Docker sandbox (direct file access)."""
    dataset = build_bucket_prediction_dataset()
    agent = build_bucket_prediction_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_bucket_local"},
    )


@task()
def citation_bucket_prediction_no_offline_prompt() -> Task:
    """Bucket prediction without the shared offline Antigravity preamble."""
    dataset = build_bucket_prediction_dataset()
    agent = build_bucket_prediction_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_bucket_no_offline"},
    )


@task()
def citation_bucket_prediction_no_offline_prompt_local() -> Task:
    """Bucket prediction (no preamble) without Docker sandbox."""
    dataset = build_bucket_prediction_dataset()
    agent = build_bucket_prediction_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_bucket_no_offline_local"},
    )


@task()
def citation_bucket_prediction_simple_task() -> Task:
    """Bucket prediction: Predict citation range (5 buckets: 0-10, 10-25, 25-60, 60-150, 150+). zeroshot"""
    dataset = build_bucket_prediction_dataset()
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are a research impact analyst. Your goal is to predict which citation range "
                "a paper will fall into based on its title and abstract.\n\n"
                "Citation buckets:\n"
                "- A: 0-10 citations (very low impact)\n"
                "- B: 10-25 citations (low impact)\n"
                "- C: 25-60 citations (moderate impact)\n"
                "- D: 60-150 citations (high impact)\n"
                "- E: 150+ citations (very high impact)\n\n"
                "Consider factors like:\n"
                "- Novelty and significance of the research problem\n"
                "- Broad applicability vs. niche focus\n"
                "- Clarity and appeal of the core contribution\n"
                "- Potential impact on the field\n"
                "- Relevance to current research trends\n\n"
                "Respond with ONLY ONE LETTER: A, B, C, D, or E\n"
                "Do not explain your reasoning. Just output the single letter corresponding to "
                "the citation range you predict."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "citation_prediction_bucket_simple"},
    )



