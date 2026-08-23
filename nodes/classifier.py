"""
nodes/classifier.py — 9B task classification with confidence + clarification.

If confidence=low and clarification_question is set, the pipeline halts
immediately and returns the question to the caller.

Base system prompt lives in config/prompts/classify.yaml.
Pinned-mode variants are handled inline (Mode B) since they require
conditional system prompt selection — the YAML system is used for the
unpinned case only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role, load_prompt, _safe_format
from pipeline.state import PipelineState
from storage.critique_store import write_run
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
    DESCRIBE = "describe"

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
    confidence: str = Field(
        default="high",
        description="high=proceed. medium=proceed with warning. low=halt and clarify."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Single specific question to resolve ambiguity. Only when confidence=low."
    )
    model_config = {"use_enum_values": True}


# ── Pinned-mode system prompts ────────────────────────────────────────────────
# Used when the caller has forced mode and/or task_type.
# The base system (from classify.yaml) is used for the unpinned case.

_SYSTEM_PINNED_MODE = """You are a task classifier for a local LLM pipeline.
The MODE has been pinned by the user — do not change it.
Determine: task_type, complexity, decompose, confidence, clarification_question.

Set confidence=low only when the task is genuinely ambiguous in a way that
would cause the wrong output. Most tasks should be high or medium confidence.
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

    # ── Choose system prompt and build messages ────────────────────────────
    if pinned_mode and pinned_task_type:
        # Mode B: both pinned — use inline system
        pin_note = f"[PINNED] mode={pinned_mode}, task_type={pinned_task_type}\n\n"
        messages = [
            {"role": "system", "content": _SYSTEM_PINNED_BOTH},
            {"role": "user",   "content": f"{pin_note}Task to classify:\n\n{task}"},
        ]
    elif pinned_mode:
        # Mode B: mode pinned — use inline system
        pin_note = f"[PINNED] mode={pinned_mode}\n\n"
        messages = [
            {"role": "system", "content": _SYSTEM_PINNED_MODE},
            {"role": "user",   "content": f"{pin_note}Task to classify:\n\n{task}"},
        ]
    else:
        # Mode A: YAML-driven (unpinned — most common path)
        messages = None

    max_attempts  = 3
    classification = None
    extra_messages: list[dict] = []

    for attempt in range(max_attempts):
        try:
            classification: TaskClassification = call_role(
                role            = "classify",
                messages        = messages,
                template_vars   = {"task": task} if messages is None else None,
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = TaskClassification,
                stage           = "classify",
                run_dir         = run_dir,
                thinking        = False,
                max_retries     = 0,
            )
            break

        except Exception as e:
            log.warning(
                "Classifier JSON validation failed (attempt %d/%d): %s",
                attempt + 1, max_attempts, str(e)
            )
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Classifier failed after {max_attempts} attempts."
                ) from e

            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed Pydantic validation:\n{str(e)}\n\n"
                        f"Please try again with strict JSON compliance."
                    ),
                },
            ]
            # For Mode B (pinned), rebuild messages without extra_messages
            # (extra_messages is appended by call_role)

    final_mode      = pinned_mode      or classification.mode
    final_task_type = pinned_task_type or classification.task_type

    log.info(
        "Classification: mode=%s type=%s complexity=%s decompose=%s confidence=%s",
        final_mode, final_task_type,
        classification.complexity, classification.decompose,
        classification.confidence,
    )

    if classification.confidence == "low" and classification.clarification_question:
        log.warning("Classifier confidence=low: %s", classification.clarification_question)

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

    classification_path = str(Path(run_dir) / "classification.json")
    Path(classification_path).write_text(
        classification.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "classification":         resolved,
        "mode":                   final_mode,
        "task_type":              final_task_type,
        "decompose":              classification.decompose,
        "classifier_confidence":  classification.confidence,
        "clarification_question": classification.clarification_question,
    }
