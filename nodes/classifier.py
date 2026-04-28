"""
nodes/classifier.py — 9B task classification with confidence + clarification.

New: if confidence=low and clarification_question is set, the pipeline
halts immediately and returns the question to the caller rather than
spending 90+ minutes on a misunderstood task.
"""

from __future__ import annotations
import logging
from clients.llm import call_role
from pipeline.state import PipelineState
from storage.critique_store import write_run
from pathlib import Path

# TaskClassification now inherits ConfidenceMixin
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

log = logging.getLogger(__name__)

class Mode(str, Enum):
    SHORT = "short"
    LONG  = "long"

class TaskType(str, Enum):
    CODING   = "coding"
    IDEATION = "ideation"
    MIXED    = "mixed"
    DESCRIBE = "describe"   # analysis/explanation/description — no implementation needed

class Complexity(str, Enum):
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"

class TaskClassification(BaseModel):
    mode:       Mode       = Field(description="short=9B executes. long=35B executes.")
    task_type:  TaskType   = Field(description="Primary nature of the task.")
    complexity: Complexity = Field(description="simple/moderate/complex.")
    decompose:  bool       = Field(description="True if >5 independent components.")
    estimated_sub_specs: Optional[int] = Field(default=None)
    reasoning:  str        = Field(description="Brief explanation of decisions.")

    # ── Confidence + clarification ────────────────────────────────────────────
    confidence: str = Field(
        default="high",
        description=(
            "high=proceed. medium=proceed with warning. "
            "low=task is ambiguous; populate clarification_question."
        )
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Single specific question to resolve ambiguity. "
            "Only populate when confidence=low. "
            "Example: 'Should this persist state across restarts?' "
            "Example: 'Is this a CLI tool or a REST API?'"
        )
    )

    model_config = {"use_enum_values": True}


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_BASE = """You are a task classifier for a local LLM pipeline.

Classify the task and return a structured JSON response.

Modes:
  short — simple, well-defined tasks; 9B executes directly.
  long  — complex, architectural, or open-ended tasks; 35B executes.

Complexity:
  simple   — single function or class, unambiguous spec.
  moderate — multi-component, some design decisions required.
  complex  — architectural, multi-file, or open-ended.

Decompose (set true when):
  - More than 5 independent components needed
  - Any single component would take >500 tokens to specify

Confidence:
  high   — task is clear; you have enough information to proceed.
  medium — minor ambiguity but you can make a reasonable assumption.
  low    — task is genuinely ambiguous; a specific clarification
           would prevent wasted computation. Populate clarification_question.

Be conservative with confidence=low — only use it when proceeding
would likely produce the wrong output. Most tasks can proceed with
high or medium confidence using reasonable assumptions.

--- FEW-SHOT EXAMPLES ---

Task: "write a python function to reverse a string"
→ mode=short, type=coding, complexity=simple, confidence=high
  reasoning: "Single function, unambiguous. 9B can handle directly."

Task: "build a rate limiter"
→ mode=long, type=coding, complexity=moderate, confidence=low
  clarification_question: "Should the rate limiter persist state across process restarts, or is in-memory only sufficient?"
  reasoning: "Persistence requirement fundamentally changes the design — SQLite vs in-memory dict."

Task: "design a microservices architecture for an e-commerce platform"
→ mode=long, type=mixed, complexity=complex, confidence=high, decompose=true, estimated_sub_specs=7
  reasoning: "Large architectural task with many independent components."

Task: "explore approaches for a distributed task queue"
→ mode=long, type=ideation, complexity=moderate, confidence=high
  reasoning: "Open-ended exploration; no implementation required."

Task: "describe the most notable features of this FEN: rnbqkb1r/..."
→ mode=short, type=describe, complexity=simple, confidence=high
  reasoning: "Asking for description/analysis, not a program. No code needed."

Task: "what is the time complexity of quicksort?"
→ mode=short, type=describe, complexity=simple, confidence=high
  reasoning: "Factual/analytical question. Answer directly, no implementation."
"""

_SYSTEM_PINNED_MODE = """You are a task classifier for a local LLM pipeline.
The MODE has been pinned by the user — do not change it.
Determine: task_type, complexity, decompose, confidence, clarification_question.

Set confidence=low and populate clarification_question only when the task is
genuinely ambiguous in a way that would cause the wrong output to be generated.
Most tasks should proceed with high or medium confidence.
"""

_SYSTEM_PINNED_BOTH = """You are a task classifier for a local LLM pipeline.
The MODE and TASK TYPE have been pinned by the user — do not change them.
Determine: complexity, decompose, confidence, clarification_question.
"""


def classify_node(state: PipelineState) -> dict:
    run_dir          = state["run_dir"]
    task             = state.get("normalised_input") or state.get("raw_text_input", "")
    pinned_mode      = state.get("mode")
    pinned_task_type = state.get("task_type")

    if pinned_mode and pinned_task_type:
        system   = _SYSTEM_PINNED_BOTH
        pin_note = f"[PINNED] mode={pinned_mode}, task_type={pinned_task_type}\n\n"
    elif pinned_mode:
        system   = _SYSTEM_PINNED_MODE
        pin_note = f"[PINNED] mode={pinned_mode}\n\n"
    else:
        system   = _SYSTEM_BASE
        pin_note = ""

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"{pin_note}Task to classify:\n\n{task}"},
    ]

    classification: TaskClassification = call_role(
        role            = "classify",
        messages        = messages,
        response_schema = TaskClassification,
        stage           = "classify",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
    )

    final_mode      = pinned_mode      or classification.mode
    final_task_type = pinned_task_type or classification.task_type

    log.info(
        "Classification: mode=%s type=%s complexity=%s decompose=%s confidence=%s",
        final_mode, final_task_type,
        classification.complexity, classification.decompose,
        classification.confidence,
    )

    if classification.confidence == "low" and classification.clarification_question:
        log.warning(
            "Classifier confidence=low: %s",
            classification.clarification_question
        )

    resolved = classification.model_copy(update={
        "mode":      final_mode,
        "task_type": final_task_type,
    })

    write_run(
        run_uuid        = state["run_uuid"],
        mode            = final_mode,
        task_type       = final_task_type,
        complexity      = classification.complexity,
        is_sub_spec     = state.get("is_sub_spec", False),
        parent_run_uuid = state.get("parent_run_uuid"),
    )

    # Write the classification to disk so we can inspect it later
    classification_path = str(Path(run_dir) / "classification.json")
    Path(classification_path).write_text(
        classification.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "classification":           resolved,
        "mode":                     final_mode,
        "task_type":                final_task_type,
        "decompose":                classification.decompose,
        "classifier_confidence":    classification.confidence,
        "clarification_question":   classification.clarification_question,
    }