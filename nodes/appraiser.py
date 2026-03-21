"""
nodes/appraiser.py — DeepCoder 14B correctness appraisal.

Receives DraftOutput + PlanSpec.
Produces AppraisalReport — consumed by bugfixer.
Thinking mode ON (RL correctness reasoning).
Does NOT generate replacement code.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.execution import AppraisalReport

log = logging.getLogger(__name__)

_SYSTEM = """You are a correctness appraisal model trained on verifiable coding problems.
Your job is to determine whether the draft implementation satisfies the PlanSpec.

DO NOT generate replacement code.
DO identify:
  - Logic errors that would cause incorrect behaviour
  - Constraint violations (requirements from the spec not satisfied)
  - Spec deltas (implementation differs from what was specified)
  - Missing requirements (things the spec required that are absent)
  - Error handling gaps
  - Type errors
  - Edge cases the spec specified that are unhandled

For each issue: state the component, severity (critical/major/minor),
category, clear description, and location if applicable.

Also flag any issues that appear to originate from imprecise ideation
(far-fetched or inconsistent assumptions baked into the implementation)
in iq2s_inherited_issues.

When your reasoning is complete, emit: <confidence>high</confidence>
"""


def appraise_node(state: PipelineState) -> dict:
    """
    Appraise the draft for correctness against the PlanSpec.
    Writes appraisal_report.json to disk.
    """
    run_dir = state["run_dir"]
    draft   = state.get("draft_output")
    plan    = state.get("plan_spec")

    if not draft or not plan:
        raise ValueError("appraise_node: missing draft_output or plan_spec")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"PlanSpec (the contract):\n{plan.model_dump_json(indent=2)}\n\n"
                f"Draft Implementation (to appraise):\n{draft.model_dump_json(indent=2)}"
            ),
        },
    ]

    report: AppraisalReport = call_role(
        role            = "appraise",
        messages        = messages,
        response_schema = AppraisalReport,
        stage           = "appraise",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = 4096,
    )

    log.info(
        "Appraisal: %s satisfaction | critical=%d major=%d minor=%d | iq2s=%d",
        report.spec_satisfaction,
        report.critical_count,
        report.major_count,
        report.minor_count,
        len(report.iq2s_inherited_issues),
    )

    # Write to disk
    appraisal_path = str(Path(run_dir) / "appraisal_report.json")
    Path(appraisal_path).write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "appraisal_report": report,
        "appraisal_path":   appraisal_path,
    }
