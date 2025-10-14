"""Inspect AI benchmark prototype for EMNLP benchmark queries.

The goal is to demonstrate a React-style agent that works entirely inside a
sandbox populated with the EMNLP accepted papers dataset. The agent is given a
small QA set and must rely on the provided tools (bash, bash session, python,
text editor, think) to surface the correct evidence from the sandbox before it
answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.tool import bash, bash_session, python, text_editor, think

SANDBOX_ROOT = Path(__file__).resolve().parent / "sandbox"
MCQ_DATASET_PATH = Path(__file__).resolve().parent / "mcq_dataset.jsonl"


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
    """Create an Inspect dataset from the MCQ JSONL file."""
    samples: List[Sample] = list(_load_samples(MCQ_DATASET_PATH))
    return MemoryDataset(samples)


def build_agent():
    """Configure a React agent confined to the EMNLP sandbox."""
    return react(
        name="emnlp-react",
        prompt=(
            "You are an research assistant working entirely inside the "
            "mounted sandbox. Use the provided tools to inspect the CSV data "
            "(notably sandbox/data/accepted_papers.csv) and any supporting "
            "files before answering. When the prompt is to select a "
            "recognition tier, choose exactly one label from Findings, Main, "
            "Outstanding, or Best. Never rely on outside knowledge."
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
def emnlp_awards_mcq_task() -> Task:
    """Multiple-choice benchmark that uses award metadata in the sandbox."""
    dataset = build_mcq_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        metadata={"benchmark": "emnlp_awards_mcq"},
    )
