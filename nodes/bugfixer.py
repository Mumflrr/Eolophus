"""
nodes/bugfixer.py — DeepCoder 14B bug fix and idiomatic review.

Receives: DraftOutput + AppraisalReport (if available) + PlanSpec.
Performs:
  1. Independent bug finding (corpus-trained pattern matching)
  2. Targeted fixes from AppraisalReport
  3. Idiomatic corrections and code quality polish
Produces: FixedOutput — what the critique ensemble receives.

Non-thinking mode — precise repair, not reasoning. Budget from routing.yaml.
Prompt lives in config/prompts/bugfix.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.guards import check_fixed_output_present
from pipeline.state import PipelineState
from schemas.execution import FixedOutput

log = logging.getLogger(__name__)


def apply_diff(original_code: str, edits: list) -> str:
    """Apply SearchReplaceBlock edits to draft code in-place."""
    modified = original_code
    for edit in edits:
        if edit.search_text in modified:
            modified = modified.replace(edit.search_text, edit.replace_text, 1)
        else:
            log.warning("Diff Engine: could not find exact search_text in code — skipping edit")
    return modified


def bugfix_node(state: PipelineState) -> dict:
    """
    Apply bug fixes and idiomatic corrections to the draft.
    Writes fixed.json to disk.
    """
    run_dir   = state["run_dir"]
    draft     = state.get("draft_output")
    appraisal = state.get("appraisal_report")
    plan      = state.get("plan_spec")
    iteration = state.get("iteration", 0)

    if not draft:
        raise ValueError("bugfix_node: missing draft_output")

    # ── Build template vars ────────────────────────────────────────────────
    appraisal_block = ""
    if appraisal:
        appraisal_block = (
            f"AppraisalReport (apply all identified fixes):\n"
            f"{appraisal.model_dump_json(indent=2)}"
        )
    else:
        appraisal_block = "[No AppraisalReport available — apply your own review only]"

    correction_block = ""
    if iteration > 0:
        verdict = state.get("validation_verdict")
        if verdict and verdict.specific_issues:
            issues_str = "\n".join(f"  - {i}" for i in verdict.specific_issues)
            correction_block = (
                f"\n[Validator feedback from iteration {iteration}]:\n{issues_str}"
            )

    fixed: FixedOutput = call_role(
        role            = "bugfix",
        template_vars   = {
            "plan_json":       plan.model_dump_json(indent=2) if plan else "not available",
            "draft_json":      draft.model_dump_json(indent=2),
            "appraisal_block": appraisal_block,
            "correction_block":correction_block,
        },
        response_schema = FixedOutput,
        stage           = "bugfix",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
    )

    # Apply diffs back onto the draft
    for fix in fixed.applied_fixes:
        for comp in draft.component_drafts:
            if comp.component_name == fix.component:
                comp.code = apply_diff(comp.code, fix.edits)

    state["draft_output"] = draft

    passed, reason = check_fixed_output_present(fixed)
    if not passed:
        log.warning("bugfix_node: output guard failed: %s", reason)

    log.info(
        "BugFix: %d applied | %d self-identified | %d unfixed | quality=%s",
        len(fixed.applied_fixes),
        len(fixed.self_identified_issues),
        len(fixed.unfixed_issues),
        fixed.overall_quality,
    )

    fixed_path = str(Path(run_dir) / "fixed.json")
    Path(fixed_path).write_text(fixed.model_dump_json(indent=2), encoding="utf-8")

    return {
        "fixed_output": fixed,
        "fixed_path":   fixed_path,
    }
