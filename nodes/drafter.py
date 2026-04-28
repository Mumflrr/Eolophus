"""
nodes/drafter.py — draft generation.

Long mode:  35B MoE UD-Q4_K_XL with two-pass strategy
            Pass 1: architectural thinking (thinking=ON, NoWait, budget_tokens)
            Pass 2: component fill (thinking=OFF)
Short mode: 9B non-thinking (rapid) or 9B thinking (careful)

Lazy evaluation guard runs after both modes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.guards import check_lazy_evaluation
from pipeline.state import PipelineState
from schemas.execution import DraftOutput, ComponentDraft
from schemas.plan_spec import PlanSpec, ComponentSpec


def _get_budget(stage: str) -> int:
    """Read thinking token budget for this stage from routing.yaml."""
    from clients.llm import _get_thinking_budget
    return _get_thinking_budget(stage)


log = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_LONG = """You are a senior software architect and implementer.
You will receive a PlanSpec and implement it completely.

PASS 1 — ARCHITECTURE (thinking mode):
Think through the overall structure, component interfaces, shared data structures,
and implementation order. Reason about edge cases and potential issues.
Reflect your confidence in the `confidence` field. If you are unsure or missing critical information, set confidence to 'low' and write a specific question to the user in `clarification_question`.

PASS 2 — IMPLEMENTATION:
Implement each component completely and correctly following the PlanSpec exactly.
Every function signature must match the spec. Every edge case must be handled.
Do not skip implementations or use placeholders."""

_SYSTEM_SHORT = """You are a senior software developer executing a straightforward task.
Implement the provided PlanSpec completely and directly.
Do not over-engineer. Follow conventions. Handle all stated edge cases.
Reflect your confidence in the `confidence` field. If you are unsure, set confidence to 'low' and write a question in `clarification_question`."""


# ── Long mode node ────────────────────────────────────────────────────────────

def draft_node(state: PipelineState) -> dict:
    """
    Generate DraftOutput from PlanSpec.
    Applies correction feedback if this is a re-draft iteration.
    """
    run_dir = state["run_dir"]
    plan    = state.get("plan_spec")
    
    if not plan:
        raise ValueError("draft_node: plan_spec missing from state")

    context = _build_correction_context(state)
    
    messages = [
        {"role": "system", "content": _SYSTEM_LONG},
        {"role": "user",   "content": f"PlanSpec:\n{plan.model_dump_json(indent=2)}\n{context}"},
    ]

    draft: DraftOutput = call_role(
        role            = "draft",
        messages        = messages,
        response_schema = DraftOutput,
        stage           = "draft",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = _get_budget("draft"),
        max_retries     = 0,
    )

    if draft.confidence == "low" and draft.clarification_question:
        log.warning("Drafter halted — needs human input: %s", draft.clarification_question)
        return {
            "pipeline_halted": True,
            "clarification_needed": draft.clarification_question
        }

    return _finalize_draft(run_dir, draft, state)


# ── Short mode node ───────────────────────────────────────────────────────────

def draft_short_node(state: PipelineState) -> dict:
    """
    9B execution for simple/moderate short-mode tasks.
    Uses thinking if complexity=moderate, skips thinking if simple.
    """
    run_dir    = state["run_dir"]
    plan       = state.get("plan_spec")
    complexity = state.get("task_complexity", "simple")
    
    if not plan:
        raise ValueError("draft_short_node: plan_spec missing")

    use_thinking = (complexity == "moderate")
    
    context = _build_correction_context(state)
    messages = [
        {"role": "system", "content": _SYSTEM_SHORT},
        {"role": "user",   "content": f"PlanSpec:\n{plan.model_dump_json(indent=2)}\n{context}"},
    ]

    draft: DraftOutput = call_role(
        role            = "draft",
        messages        = messages,
        response_schema = DraftOutput,
        stage           = "draft",
        run_dir         = run_dir,
        thinking        = use_thinking,
        budget_tokens   = 2048 if use_thinking else 0,
        max_retries     = 0,
    )

    if draft.confidence == "low" and draft.clarification_question:
        log.warning("Drafter (short) halted — needs human input: %s", draft.clarification_question)
        return {
            "pipeline_halted": True,
            "clarification_needed": draft.clarification_question
        }

    return _finalize_draft(run_dir, draft, state)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finalize_draft(run_dir: str, draft: DraftOutput, state: PipelineState) -> dict:
    """
    Run lazy evaluation guard.
    Write draft to disk (overwritten on re-draft).
    Return updated state fields.
    """
    passed, reason = check_lazy_evaluation(draft)

    log.info(
        "Draft: %d components | guard=%s%s",
        len(draft.component_drafts),
        "pass" if passed else "FAIL",
        f" ({reason})" if not passed else "",
    )

    # Write to disk even if guard failed (for debugging)
    draft_path = str(Path(run_dir) / "draft.json")
    Path(draft_path).write_text(
        draft.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "draft_output":  draft,
        "draft_path":    draft_path,
        "_guard_passed": passed,
        "_guard_reason": reason,
    }


def _build_correction_context(state: PipelineState) -> str:
    """Build correction feedback string for re-draft iterations."""
    iteration = state.get("iteration", 0)
    if iteration == 0:
        return ""

    verdict = state.get("validation_verdict")
    if not verdict:
        return ""

    parts = [f"[REDRAFT — iteration {iteration}]"]
    if verdict.description:
        parts.append(f"Issue: {verdict.description}")
    if verdict.specific_issues:
        parts.append("Specific issues to fix:")
        for issue in verdict.specific_issues:
            parts.append(f"  - {issue}")
            
    return "\n" + "\n".join(parts)