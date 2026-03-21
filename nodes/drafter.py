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

log = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_LONG = """You are a senior software architect and implementer.
You will receive a PlanSpec and implement it completely.

PASS 1 — ARCHITECTURE (thinking mode):
Think through the overall structure, component interfaces, shared data structures,
and implementation order. Reason about edge cases and potential issues.
When you are confident in the architecture, emit: <confidence>high</confidence>

PASS 2 — IMPLEMENTATION:
Implement each component completely and correctly following the PlanSpec exactly.
Every function signature must match the spec. Every edge case must be handled.
Do not skip implementation or write placeholder code.
"""

_SYSTEM_SHORT = """You are implementing a software task from a structured plan.
Implement all components completely and correctly.
Follow the PlanSpec exactly — every function signature, every edge case.
Do not write placeholder code or skip implementations."""


# ── Long mode: 35B draft ──────────────────────────────────────────────────────

def draft_node(state: PipelineState) -> dict:
    """Long mode draft — 35B MoE with architecture thinking pass."""
    run_dir   = state["run_dir"]
    plan      = state.get("plan_spec")
    iteration = state.get("iteration", 0)

    if not plan:
        raise ValueError("draft_node: no plan_spec in state")

    plan_json = plan.model_dump_json(indent=2)

    # Build correction context if re-drafting
    correction_ctx = _build_correction_context(state)

    user_content = f"Implement this PlanSpec completely:\n\n{plan_json}"
    if correction_ctx:
        user_content = f"{correction_ctx}\n\n{user_content}"

    messages = [
        {"role": "system", "content": _SYSTEM_LONG},
        {"role": "user",   "content": user_content},
    ]

    # Architecture pass: thinking ON with NoWait + budget
    draft: DraftOutput = call_role(
        role            = "draft_long",
        messages        = messages,
        response_schema = DraftOutput,
        stage           = "draft",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = 8192,
    )

    return _finalise_draft(draft, run_dir, state)


# ── Short mode: 9B draft ──────────────────────────────────────────────────────

def draft_short_node(state: PipelineState) -> dict:
    """Short mode draft — 9B (non-thinking rapid or thinking careful)."""
    run_dir       = state["run_dir"]
    plan          = state.get("plan_spec")
    classification = state.get("classification")

    if not plan:
        raise ValueError("draft_short_node: no plan_spec in state")

    # Careful short mode uses thinking; rapid does not
    use_thinking = False
    role         = "draft_short"

    plan_json = plan.model_dump_json(indent=2)
    correction_ctx = _build_correction_context(state)

    user_content = f"Implement this plan completely:\n\n{plan_json}"
    if correction_ctx:
        user_content = f"{correction_ctx}\n\n{user_content}"

    messages = [
        {"role": "system", "content": _SYSTEM_SHORT},
        {"role": "user",   "content": user_content},
    ]

    draft: DraftOutput = call_role(
        role            = role,
        messages        = messages,
        response_schema = DraftOutput,
        stage           = "draft_short",
        run_dir         = run_dir,
        thinking        = use_thinking,
        budget_tokens   = 2048 if use_thinking else 0,
    )

    return _finalise_draft(draft, run_dir, state)


# ── Shared finalisation ───────────────────────────────────────────────────────

def _finalise_draft(draft: DraftOutput, run_dir: str, state: PipelineState) -> dict:
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
    if verdict.suggested_fix_direction:
        parts.append(f"Direction: {verdict.suggested_fix_direction}")

    return "\n".join(parts)
