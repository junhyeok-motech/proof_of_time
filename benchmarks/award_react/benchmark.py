"""Inspect AI benchmark for award prediction tasks.

This benchmark evaluates LLMs' ability to classify research papers into
recognition tiers (Best/Outstanding/Main/Findings) by analyzing historical
award-winning papers in a sandboxed environment.

Datasets:
- pre-cutoff_mcq.jsonl: Pre-2025 conference awards (ACL/EMNLP/NAACL 2018-2024)
- post-cutoff_emnlp.jsonl: Post-2025 EMNLP awards
- post-cutoff_acl_naacl.jsonl: Post-2025 ACL/NAACL awards
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import match
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
PRE_CUTOFF_DATASET = Path(__file__).resolve().parent / "pre-cutoff_mcq.jsonl"
POST_CUTOFF_EMNLP_DATASET = Path(__file__).resolve().parent / "post-cutoff_emnlp.jsonl"
POST_CUTOFF_ACL_NAACL_DATASET = Path(__file__).resolve().parent / "post-cutoff_acl_naacl.jsonl"


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


def _load_dataset(path: Path) -> Dataset:
    """Create an Inspect dataset from a JSONL file."""
    samples: List[Sample] = list(_load_samples(path))
    return MemoryDataset(samples)


def build_agent(use_offline_prompt: bool = True):
    """Configure a React agent for award prediction in sandbox."""
    offline_prefix = f"{get_offline_preamble()}\n\n" if use_offline_prompt else ""
    return react(
        name="award-prediction-react",
        prompt=(
            f"{offline_prefix}"
            "You are a research paper quality assessor. Your task is to classify papers into recognition tiers "
            "by comparing them to previous award-winning and accepted papers.\n\n"
            "Recognition tiers:\n"
            "- Best: Best Paper Award winners (groundbreaking, paradigm-shifting work)\n"
            "- Outstanding: Outstanding Paper Awards (exceptional contributions)\n"
            "- Main: Main conference track (solid accepted papers)\n"
            "- Findings: Findings track (good work, below main bar)\n\n"
            "PROCESS:\n"
            "1. Use python() to read and analyze sandbox/data/accepted_papers.csv\n"
            "2. Look at papers with 'Best' or 'Outstanding' awards to understand patterns\n"
            "3. Compare the given paper's novelty, impact, and quality to these examples\n"
            "4. Make your classification decision\n"
            "5. Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings\n\n"
            "CRITICAL: After using tools to investigate, provide your final answer as a SINGLE WORD on its own line. "
            "Do not add explanations after your answer. The answer must be exactly one of: Best, Outstanding, Main, Findings"
        ),
        tools=[
            think(),
            python(),
            bash(),
            bash_session(),
            text_editor(),
        ],
    )


# =============================================================================
# Pre-cutoff tasks (historical data before 2025)
# =============================================================================

@task()
def pre_cutoff_task() -> Task:
    """Pre-cutoff benchmark - ReAct agent classifies papers using historical award data."""
    dataset = _load_dataset(PRE_CUTOFF_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_pre_cutoff"},
    )


@task()
def pre_cutoff_task_local() -> Task:
    """Pre-cutoff benchmark without Docker sandbox (direct file access)."""
    dataset = _load_dataset(PRE_CUTOFF_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_pre_cutoff_local"},
    )


@task()
def pre_cutoff_task_no_offline_prompt() -> Task:
    """Pre-cutoff benchmark without the shared offline Antigravity preamble."""
    dataset = _load_dataset(PRE_CUTOFF_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_pre_cutoff_no_offline"},
    )


@task()
def pre_cutoff_task_no_offline_prompt_local() -> Task:
    """Pre-cutoff benchmark (no preamble) without Docker sandbox."""
    dataset = _load_dataset(PRE_CUTOFF_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_pre_cutoff_no_offline_local"},
    )


@task()
def pre_cutoff_simple_task() -> Task:
    """Simple pre-cutoff benchmark - direct generation without tools."""
    dataset = _load_dataset(PRE_CUTOFF_DATASET)
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are an expert at classifying research papers into conference recognition tiers. "
                "Given a paper's title and abstract, determine which tier it belongs to:\n\n"
                "- **Best**: Best Paper Award winners (groundbreaking, top 0.1% contributions)\n"
                "- **Outstanding**: Outstanding Paper Award (exceptional quality, top 1%)\n"
                "- **Main**: Main conference track (high quality, accepted papers)\n"
                "- **Findings**: Findings track (good work, didn't meet main conference bar)\n\n"
                "Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings.\n"
                "Do not explain your reasoning. Just output the single tier name."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "award_pre_cutoff_simple"},
    )


# =============================================================================
# Post-cutoff EMNLP tasks (2025+ EMNLP papers)
# =============================================================================

@task()
def post_cutoff_emnlp_task() -> Task:
    """Post-cutoff EMNLP benchmark - ReAct agent classifies 2025+ EMNLP papers."""
    dataset = _load_dataset(POST_CUTOFF_EMNLP_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_emnlp"},
    )


@task()
def post_cutoff_emnlp_task_local() -> Task:
    """Post-cutoff EMNLP benchmark without Docker sandbox."""
    dataset = _load_dataset(POST_CUTOFF_EMNLP_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_emnlp_local"},
    )


@task()
def post_cutoff_emnlp_task_no_offline_prompt() -> Task:
    """Post-cutoff EMNLP benchmark without the shared offline Antigravity preamble."""
    dataset = _load_dataset(POST_CUTOFF_EMNLP_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_emnlp_no_offline"},
    )


@task()
def post_cutoff_emnlp_task_no_offline_prompt_local() -> Task:
    """Post-cutoff EMNLP benchmark (no preamble) without Docker sandbox."""
    dataset = _load_dataset(POST_CUTOFF_EMNLP_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_emnlp_no_offline_local"},
    )


@task()
def post_cutoff_emnlp_simple_task() -> Task:
    """Simple post-cutoff EMNLP benchmark - direct generation without tools."""
    dataset = _load_dataset(POST_CUTOFF_EMNLP_DATASET)
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are an expert at classifying research papers into conference recognition tiers. "
                "Given a paper's title and abstract, determine which tier it belongs to:\n\n"
                "- **Best**: Best Paper Award winners (groundbreaking, top 0.1% contributions)\n"
                "- **Outstanding**: Outstanding Paper Award (exceptional quality, top 1%)\n"
                "- **Main**: Main conference track (high quality, accepted papers)\n"
                "- **Findings**: Findings track (good work, didn't meet main conference bar)\n\n"
                "Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings.\n"
                "Do not explain your reasoning. Just output the single tier name."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_emnlp_simple"},
    )


# =============================================================================
# Post-cutoff ACL/NAACL tasks (2025+ ACL/NAACL papers)
# =============================================================================

@task()
def post_cutoff_acl_naacl_task() -> Task:
    """Post-cutoff ACL/NAACL benchmark - ReAct agent classifies 2025+ ACL/NAACL papers."""
    dataset = _load_dataset(POST_CUTOFF_ACL_NAACL_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_acl_naacl"},
    )


@task()
def post_cutoff_acl_naacl_task_local() -> Task:
    """Post-cutoff ACL/NAACL benchmark without Docker sandbox."""
    dataset = _load_dataset(POST_CUTOFF_ACL_NAACL_DATASET)
    agent = build_agent()
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_acl_naacl_local"},
    )


@task()
def post_cutoff_acl_naacl_task_no_offline_prompt() -> Task:
    """Post-cutoff ACL/NAACL benchmark without the shared offline Antigravity preamble."""
    dataset = _load_dataset(POST_CUTOFF_ACL_NAACL_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox="docker",
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_acl_naacl_no_offline"},
    )


@task()
def post_cutoff_acl_naacl_task_no_offline_prompt_local() -> Task:
    """Post-cutoff ACL/NAACL benchmark (no preamble) without Docker sandbox."""
    dataset = _load_dataset(POST_CUTOFF_ACL_NAACL_DATASET)
    agent = build_agent(use_offline_prompt=False)
    return Task(
        dataset=dataset,
        solver=agent,
        scorer=match(),
        sandbox=None,
        config=AGENT_TASK_CONFIG,
        **AGENT_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_acl_naacl_no_offline_local"},
    )


@task()
def post_cutoff_acl_naacl_simple_task() -> Task:
    """Simple post-cutoff ACL/NAACL benchmark - direct generation without tools."""
    dataset = _load_dataset(POST_CUTOFF_ACL_NAACL_DATASET)
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                f"{get_offline_preamble()}\n\n"
                "You are an expert at classifying research papers into conference recognition tiers. "
                "Given a paper's title and abstract, determine which tier it belongs to:\n\n"
                "- **Best**: Best Paper Award winners (groundbreaking, top 0.1% contributions)\n"
                "- **Outstanding**: Outstanding Paper Award (exceptional quality, top 1%)\n"
                "- **Main**: Main conference track (high quality, accepted papers)\n"
                "- **Findings**: Findings track (good work, didn't meet main conference bar)\n\n"
                "Respond with ONLY ONE WORD: Best, Outstanding, Main, or Findings.\n"
                "Do not explain your reasoning. Just output the single tier name."
            ),
            generate(),
        ],
        scorer=match(),
        config=SIMPLE_TASK_CONFIG,
        **SIMPLE_TASK_LIMITS,
        metadata={"benchmark": "award_post_cutoff_acl_naacl_simple"},
    )
