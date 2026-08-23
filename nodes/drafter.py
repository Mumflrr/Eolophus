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

from clients.llm import call_role
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
            )
            break
        except Exception as e:
            log.warning("Draft JSON validation failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Drafter failed after {max_attempts} attempts.") from e
            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {"role": "user", "content": f"Pydantic validation error:\n{e}\n\nPlease output valid JSON."},
            ]

    if draft.confidence == "low" and draft.clarification_question:
        log.warning("Drafter halted — needs human input: %s", draft.clarification_question)
        return {
            "pipeline_halted":      True,
            "clarification_needed": draft.clarification_question,
        }

    return _finalize_draft(run_dir, draft, state)


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
    )

    if draft.confidence == "low" and draft.clarification_question:
        log.warning("Drafter (short) halted — needs human input: %s", draft.clarification_question)
        return {
            "pipeline_halted":      True,
            "clarification_needed": draft.clarification_question,
        }

    return _finalize_draft(run_dir, draft, state)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finalize_draft(run_dir: str, draft: DraftOutput, state: PipelineState) -> dict:
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

    draft_path = str(Path(run_dir) / "draft.json")
    Path(draft_path).write_text(draft.model_dump_json(indent=2), encoding="utf-8")

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
