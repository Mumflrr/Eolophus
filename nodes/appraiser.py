"""
nodes/appraiser.py — DeepCoder 14B correctness appraisal.

Receives DraftOutput + PlanSpec.
Produces AppraisalReport — consumed by bugfixer.
Thinking mode ON (RL correctness reasoning). Budget from routing.yaml.
Does NOT generate replacement code.

Prompt lives in config/prompts/appraise.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.execution import AppraisalReport

log = logging.getLogger(__name__)


def appraise_node(state: PipelineState) -> dict:
    """
    Generate AppraisalReport from DraftOutput.
    Evaluates execution correctness against the PlanSpec.
    Writes appraisal_report.json to disk.
    """
    run_dir = state["run_dir"]
    draft   = state.get("draft_output")
    plan    = state.get("plan_spec")

    if not draft or not plan:
        raise ValueError("appraise_node: missing draft_output or plan_spec")

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("appraise")

    report: AppraisalReport = call_role(
        role            = "appraise",
        template_vars   = {
            "plan_json":  plan.model_dump_json(indent=2),
            "draft_json": draft.model_dump_json(indent=2),
        },
        response_schema = AppraisalReport,
        stage           = "appraise",
        run_dir         = run_dir,
        thinking        = True,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(report, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(report, "_escalated_from", None)
        escalated_models["appraise"] = escalated_to_attr
        escalation_history.append({
            "stage":      "appraise",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "low_confidence" if report.confidence != "low" else "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Appraise escalated %s → %s", escalated_from_attr, escalated_to_attr)

    human_in_the_loop = state.get("human_in_the_loop", True)
    if report.confidence == "low" and report.clarification_question:
        if human_in_the_loop:
            log.warning("Appraiser halted — needs human input: %s", report.clarification_question)
            return {
                "pipeline_halted":      True,
                "clarification_needed": report.clarification_question,
                "escalated_models":     escalated_models,
                "escalation_history":   escalation_history,
            }
        log.warning(
            "Appraiser confidence=low after escalation exhausted, but "
            "human_in_the_loop=False (set-and-forget) — proceeding "
            "best-effort. Original question was: %s",
            report.clarification_question,
        )

    log.info(
        "Appraisal: %s satisfaction | critical=%d major=%d minor=%d | iq2s=%d",
        report.spec_satisfaction,
        report.critical_count,
        report.major_count,
        report.minor_count,
        len(report.iq2s_inherited_issues),
    )

    appraisal_path = str(Path(run_dir) / "appraisal_report.json")
    Path(appraisal_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return {
        "appraisal_report":   report,
        "appraisal_path":     appraisal_path,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }