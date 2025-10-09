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
QA_DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset.jsonl"


def _load_samples(path: Path) -> Iterable[Sample]:
    """Parse the local QA dataset into Inspect `Sample` objects."""
    sandbox_mount = str(SANDBOX_ROOT.resolve())
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            payload = json.loads(line)
            yield Sample(
                id=idx,
                input=payload["question"],
                target=payload["answer"],
                metadata=payload.get("metadata", {}),
                files={"": sandbox_mount},
            )


def build_dataset() -> Dataset:
    """Create an Inspect dataset from the QA JSONL file."""
    samples: List[Sample] = list(_load_samples(QA_DATASET_PATH))
    return MemoryDataset(samples)


def build_agent():
    """Configure a React agent confined to the EMNLP sandbox."""
    return react(
        name="emnlp-react",
        prompt=(
            "You are an EMNLP research assistant. Answer the user's question "
            "using only the resources stored in the mounted sandbox. Inspect "
            "the JSONL dataset and any supporting files via the provided tools "
            "before producing a final answer. Never rely on outside knowledge."
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
def emnlp_react_task() -> Task:
    """Benchmark entry point registered with Inspect."""
    dataset = build_dataset()
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        metadata={"benchmark": "emnlp_react"},
    )
