"""
nodes/bugfixer.py — Qwen2.5 Coder 14B bug fix and idiomatic review.

Receives: DraftOutput + AppraisalReport (if available) + PlanSpec.
Performs:
  1. Independent bug finding (corpus-trained pattern matching)
  2. Targeted fixes from AppraisalReport
  3. Idiomatic corrections and code quality polish
Produces: FixedOutput — what the critique ensemble receives.

Non-thinking mode — precise repair, not reasoning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.guards import check_fixed_output_present
from pipeline.state import PipelineState
from schemas.execution import FixedOutput

log = logging.getLogger(__name__)

_SYSTEM = """You are a code repair and review specialist with deep knowledge of
real-world software patterns and library conventions.

You will receive a draft implementation, an optional appraisal report identifying
issues, and the original plan spec.

Your job:
1. Apply ALL fixes identified in the AppraisalReport (if present).
2. Independently review the code for bugs your expertise identifies
   that the appraisal may have missed — focus on: incorrect API usage,
   non-idiomatic patterns, missing imports, subtle runtime errors,
   incomplete error handling, type mismatches.
3. Polish for real-world code quality: naming, structure, conventions.

Do NOT rewrite working code without a reason.
Do NOT change interfaces specified in the PlanSpec.
Record every change you make in applied_fixes.
Record any issues you found independently in self_identified_issues.
If an appraisal issue cannot be fixed at this stage, note it in unfixed_issues.
"""


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

    # Build user message
    parts = []

    if plan:
        parts.append(f"PlanSpec (interfaces must be preserved):\n{plan.model_dump_json(indent=2)}")

    parts.append(f"\nDraft to fix:\n{draft.model_dump_json(indent=2)}")

    if appraisal:
        parts.append(f"\nAppraisalReport (apply all identified fixes):\n{appraisal.model_dump_json(indent=2)}")
    else:
        parts.append("\n[No AppraisalReport available — apply your own review only]")

    # Include validator feedback if this is a correction-loop re-fix
    if iteration > 0:
        verdict = state.get("validation_verdict")
        if verdict and verdict.specific_issues:
            issues_str = "\n".join(f"  - {i}" for i in verdict.specific_issues)
            parts.append(f"\n[Validator feedback from previous iteration]:\n{issues_str}")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": "\n\n".join(parts)},
    ]

    fixed: FixedOutput = call_role(
        role            = "bugfix",
        messages        = messages,
        response_schema = FixedOutput,
        stage           = "bugfix",
        run_dir         = run_dir,
        thinking        = False,
    )

    # Guard: verify output was actually produced
    passed, reason = check_fixed_output_present(fixed)
    if not passed:
        log.warning("bugfix_node: output guard failed: %s", reason)

    log.info(
        "BugFix: %d applied, %d self-identified, %d unfixed | quality=%s",
        len(fixed.applied_fixes),
        len(fixed.self_identified_issues),
        len(fixed.unfixed_issues),
        fixed.overall_quality,
    )

    # Write to disk
    fixed_path = str(Path(run_dir) / "fixed.json")
    Path(fixed_path).write_text(
        fixed.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "fixed_output": fixed,
        "fixed_path":   fixed_path,
    }
