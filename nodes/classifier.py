"""
nodes/classifier.py — 9B task classification.

Produces TaskClassification determining mode, task_type, complexity,
and whether to decompose into sub-specs.
Runs non-thinking (fast classification, not reasoning).
"""

from __future__ import annotations

import logging

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.task_classification import TaskClassification
from storage.critique_store import write_run

log = logging.getLogger(__name__)

_SYSTEM = """You are a task classifier for a local LLM pipeline.
Analyse the task and return a structured classification.

Modes:
  short — simple, well-defined tasks; interactive speed; 9B executes.
  long  — complex, architectural, or open-ended tasks; 35B executes; ideation fires for mixed/ideation types.

Decompose rules (set decompose=true when):
  - More than 5 independent components in the plan
  - Any single component would require more than ~500 tokens to specify

Be conservative with 'long' mode — only use it when the task genuinely
requires architectural reasoning or multi-file/multi-component work.
"""


def classify_node(state: PipelineState) -> dict:
    """
    Classify the task. Writes run entry to database.
    Returns updated state fields.
    """
    run_dir = state["run_dir"]
    task    = state.get("normalised_input") or state.get("raw_text_input", "")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"Task to classify:\n\n{task}"},
    ]

    classification: TaskClassification = call_role(
        role            = "classify",
        messages        = messages,
        response_schema = TaskClassification,
        stage           = "classify",
        run_dir         = run_dir,
        thinking        = False,
    )

    log.info(
        "Classification: mode=%s type=%s complexity=%s decompose=%s",
        classification.mode, classification.task_type,
        classification.complexity, classification.decompose,
    )

    # Write run entry to database
    write_run(
        run_uuid        = state["run_uuid"],
        mode            = classification.mode,
        task_type       = classification.task_type,
        complexity      = classification.complexity,
        is_sub_spec     = state.get("is_sub_spec", False),
        parent_run_uuid = state.get("parent_run_uuid"),
    )

    return {
        "classification": classification,
        "mode":           classification.mode,
        "task_type":      classification.task_type,
        "decompose":      classification.decompose,
    }
