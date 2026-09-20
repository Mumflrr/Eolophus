"""
nodes/drafter.py — draft generation.

Long mode:  35B MoE with thinking (architectural reasoning + code in one pass).
            Budget from routing.yaml draft stage.
Short mode: 9B — thinking for moderate complexity, non-thinking for simple.
            Budget from routing.yaml draft stage.

Prompts live in config/prompts/draft.yaml and draft_short.yaml.
Lazy evaluation and AST syntax guards run after both modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from clients.llm import call_role, call_role_with_repair, write_iteration_artifact
from clients.tools import format_search_context
from nodes._shared import check_confidence_halt, record_escalation
from pipeline.guards import check_lazy_evaluation, check_ast_syntax
from pipeline.state import PipelineState
from schemas.execution import DraftOutput

log = logging.getLogger(__name__)


# ── Long mode node ────────────────────────────────────────────────────────────

def draft_node(state: PipelineState) -> dict:
    """35B draft generation for long-mode tasks."""
    run_dir = state["run_dir"]
    plan    = state.get("plan_spec")

    if not plan:
        raise ValueError("draft_node: plan_spec missing from state")

    correction_block = _build_correction_context(state) + _build_search_context(state)
    dense_plan       = plan.model_dump_json(exclude_none=True)

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("draft")

    draft: DraftOutput = call_role_with_repair(
        role            = "draft",
        repair_hint     = "Please output valid JSON.",
        template_vars   = {
            "plan_json":        dense_plan,
            "correction_block": correction_block,
        },
        response_schema = DraftOutput,
        stage           = "draft",
        run_dir         = run_dir,
        thinking        = True,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models, escalation_history = record_escalation(state, "draft", draft)
    halt = check_confidence_halt(state, draft, "Drafter (draft)", escalated_models, escalation_history)
    if halt:
        return halt

    return _finalize_draft(run_dir, draft, state, escalated_models, escalation_history)


# ── Short mode node ───────────────────────────────────────────────────────────

def draft_short_node(state: PipelineState) -> dict:
    """9B execution for simple/moderate short-mode tasks."""
    run_dir    = state["run_dir"]
    plan       = state.get("plan_spec")
    complexity = state.get("task_complexity", "simple")

    if not plan:
        raise ValueError("draft_short_node: plan_spec missing")

    use_thinking     = (complexity == "moderate")
    correction_block = _build_correction_context(state) + _build_search_context(state)
    dense_plan       = plan.model_dump_json(exclude_none=True)

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("draft_short")

    draft: DraftOutput = call_role(
        role            = "draft_short",
        template_vars   = {
            "plan_json":        dense_plan,
            "correction_block": correction_block,
        },
        response_schema = DraftOutput,
        stage           = "draft",
        run_dir         = run_dir,
        thinking        = use_thinking,
        budget_tokens   = 2048 if use_thinking else 0,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models, escalation_history = record_escalation(state, "draft_short", draft)
    halt = check_confidence_halt(state, draft, "Drafter (draft_short)", escalated_models, escalation_history)
    if halt:
        return halt

    return _finalize_draft(run_dir, draft, state, escalated_models, escalation_history)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finalize_draft(
    run_dir: str, draft: DraftOutput, state: PipelineState,
    escalated_models: Optional[dict] = None,
    escalation_history: Optional[list] = None,
) -> dict:
    """Run lazy evaluation and AST syntax guards. Write draft to disk."""
    passed, reason = check_lazy_evaluation(draft)

    ast_failures = 0
    for cd in draft.component_drafts:
        if cd.code and ("def " in cd.code or "class " in cd.code):
            ast_passed, ast_reason = check_ast_syntax(cd.code)
            if not ast_passed:
                ast_failures += 1
                cd.notes = (cd.notes or "") + f"\n\n[SYSTEM AST GUARD FAILED]:\n{ast_reason}"

    log.info(
        "Draft: %d components | lazy_guard=%s | ast_failures=%d",
        len(draft.component_drafts),
        "pass" if passed else f"FAIL ({reason})",
        ast_failures,
    )

    draft_path = write_iteration_artifact(
        run_dir, "draft.json", draft.model_dump_json(indent=2), state.get("iteration", 0),
    )

    result = {
        "draft_output":  draft,
        "draft_path":    draft_path,
        "_guard_passed": passed,
        "_guard_reason": reason,
    }
    if escalated_models is not None:
        result["escalated_models"] = escalated_models
    if escalation_history is not None:
        result["escalation_history"] = escalation_history
    return result


def _build_search_context(state: PipelineState) -> str:
    """
    Web-search context for the draft prompt (see clients/tools.py's
    format_search_context). "" when use_search is off.

    Delivered by appending to the existing {correction_block} template var
    rather than a new {search_block} slot, so this works without touching
    config/prompts/draft.yaml / draft_short.yaml. If you'd rather have it
    in its own labelled spot: add {search_block} to those two user_templates
    and pass it as its own template var in the two call_role calls above
    (always — "" when unused — or the literal placeholder leaks into the
    prompt), instead of appending here.
    """
    return format_search_context(state.get("use_search"), state.get("search_notes"))


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