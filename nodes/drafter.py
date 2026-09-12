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

from clients.llm import call_role, TruncatedOutputError, write_iteration_artifact
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

    correction_block = _build_correction_context(state)
    dense_plan       = plan.model_dump_json(exclude_none=True)

    extra_messages: list[dict] = []
    draft = None
    max_attempts = 3

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("draft")

    for attempt in range(max_attempts):
        try:
            draft: DraftOutput = call_role(
                role            = "draft",
                template_vars   = {
                    "plan_json":        dense_plan,
                    "correction_block": correction_block,
                },
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = DraftOutput,
                stage           = "draft",
                run_dir         = run_dir,
                thinking        = True,
                max_retries     = 0,
                profile         = profile,
                current_model_override = current_model_override,
            )
            break
        except TruncatedOutputError:
            # See classifier.py's identical guard — a token-cap
            # truncation isn't a JSON validation failure, so retrying
            # with the same cap plus a "please output valid JSON" nudge
            # won't help, and wrapping it in RuntimeError below would
            # hide it from pipeline/graph.py's generic truncation-retry
            # wrapper. Propagate immediately instead.
            raise
        except Exception as e:
            log.warning("Draft JSON validation failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Drafter failed after {max_attempts} attempts.") from e
            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {"role": "user", "content": f"Pydantic validation error:\n{e}\n\nPlease output valid JSON."},
            ]

    escalated_models, escalation_history, halt = _apply_escalation_and_confidence(
        state, draft, "draft",
    )
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
    correction_block = _build_correction_context(state)
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

    escalated_models, escalation_history, halt = _apply_escalation_and_confidence(
        state, draft, "draft_short",
    )
    if halt:
        return halt

    return _finalize_draft(run_dir, draft, state, escalated_models, escalation_history)


# ── Escalation + confidence helper ────────────────────────────────────────────
# Shared by draft_node and draft_short_node — same pattern as classifier.py/
# planner.py's inline versions, factored out here since drafter.py has two
# call sites that both need it. See design doc §2.3/§2.6.

def _apply_escalation_and_confidence(
    state: PipelineState, draft: DraftOutput, stage_key: str,
) -> tuple[dict, list, Optional[dict]]:
    """
    Returns (escalated_models, escalation_history, halt_dict_or_None).
    If halt_dict_or_None is not None, the caller should return it
    immediately instead of proceeding to _finalize_draft.
    """
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(draft, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(draft, "_escalated_from", None)
        escalated_models[stage_key] = escalated_to_attr
        escalation_history.append({
            "stage":      stage_key,
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "low_confidence" if draft.confidence != "low" else "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("%s escalated %s → %s", stage_key, escalated_from_attr, escalated_to_attr)

    human_in_the_loop = state.get("human_in_the_loop", True)
    if draft.confidence == "low" and draft.clarification_question:
        if human_in_the_loop:
            log.warning("Drafter (%s) halted — needs human input: %s",
                        stage_key, draft.clarification_question)
            return escalated_models, escalation_history, {
                "pipeline_halted":      True,
                "clarification_needed": draft.clarification_question,
                "escalated_models":     escalated_models,
                "escalation_history":   escalation_history,
            }
        log.warning(
            "Drafter (%s) confidence=low after escalation exhausted, but "
            "human_in_the_loop=False (set-and-forget) — proceeding "
            "best-effort. Original question was: %s",
            stage_key, draft.clarification_question,
        )

    return escalated_models, escalation_history, None


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