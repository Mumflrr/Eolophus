"""
nodes/_shared.py — small helpers used by most node files.

Every node that calls call_role() needs the same two bits of bookkeeping
afterward: record whether the call escalated to a bigger model, and (for
the confidence-bearing schemas) decide whether a low-confidence result
should halt the pipeline for human input. Both were previously copy-pasted
into classifier.py, planner.py, appraiser.py, bugfixer.py, critics.py,
distiller.py, drafter.py, ideation.py, validator.py, and vision.py.

Task tagging (derive_task_tags) had a similar problem: three independent
keyword lists (planner.py, critics.py, distiller.py) that had already
drifted out of sync with each other.
"""

from __future__ import annotations

import logging
from typing import Optional

from pipeline.state import PipelineState

log = logging.getLogger(__name__)


def record_escalation(
    state:      PipelineState,
    stage_key:  str,
    result,
    trigger:    Optional[str] = None,
) -> tuple[dict, list]:
    """
    If `result` carries call_role()'s _escalated_to/_escalated_from
    attributes (see call_role's docstring), append an entry to
    escalation_history and update escalated_models. Always returns fresh
    copies of both — callers put these straight into their return dict.

    trigger: pass "low_confidence" or "truncation" explicitly when the
    caller knows which one occurred (e.g. classifier.py/planner.py, which
    see the raw EscalationNeeded). Left as None for schemas with no
    confidence field — call_model_with_tools' escalation path is only
    ever entered on TruncatedOutputError for those, so "truncation" is
    correct, not a guess.
    """
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])

    escalated_to = getattr(result, "_escalated_to", None)
    if not escalated_to:
        return escalated_models, escalation_history

    escalated_from = getattr(result, "_escalated_from", None)
    escalated_models[stage_key] = escalated_to
    escalation_history.append({
        "stage":      stage_key,
        "from_model": escalated_from,
        "to_model":   escalated_to,
        "trigger":    trigger or "truncation",
        "iteration":  state.get("iteration", 0),
    })
    log.info("%s escalated %s → %s", stage_key, escalated_from, escalated_to)
    return escalated_models, escalation_history


def check_confidence_halt(
    state:              PipelineState,
    result,
    node_label:         str,
    escalated_models:   dict,
    escalation_history: list,
) -> Optional[dict]:
    """
    For schemas with a confidence field: if confidence=="low" and the
    model asked a clarification_question, either halt the pipeline
    (human_in_the_loop=True, the default) or log and let the caller
    proceed best-effort ("set-and-forget" mode).

    Returns the halt dict to return immediately, or None to proceed.
    """
    if result.confidence != "low" or not result.clarification_question:
        return None

    if state.get("human_in_the_loop", True):
        log.warning("%s halted — needs human input: %s", node_label, result.clarification_question)
        return {
            "pipeline_halted":      True,
            "clarification_needed": result.clarification_question,
            "escalated_models":     escalated_models,
            "escalation_history":   escalation_history,
        }

    log.warning(
        "%s confidence=low after escalation exhausted, but human_in_the_loop=False "
        "(set-and-forget) — proceeding best-effort. Original question was: %s",
        node_label, result.clarification_question,
    )
    return None


# Keywords used to tag a task for lesson storage/retrieval. planner.py's
# list was the superset of the three that previously existed separately;
# critics.py's and distiller.py's had already drifted to different,
# smaller subsets of this.
_TAG_KEYWORDS = [
    "python", "fastapi", "async", "django", "typescript", "react",
    "database", "rest", "docker", "testing", "pydantic", "sqlalchemy",
    "cli", "class", "error_handling",
]


def derive_task_tags(task_type: str, text: str) -> list[str]:
    """Tag a task with task_type plus any _TAG_KEYWORDS found in `text`
    (raw task description, or plan.moe_routing_context — either works,
    this only does substring matching on lowercased text)."""
    tags = [task_type]
    lowered = (text or "").lower()
    tags.extend(kw for kw in _TAG_KEYWORDS if kw in lowered)
    return list(set(tags))