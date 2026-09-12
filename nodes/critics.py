"""
nodes/critics.py — critique ensemble.

Critic A: 9B non-thinking — coherence and completeness.
Critic B: DeepCoder 14B thinking — logical correctness verification.

Critics run independently. Neither sees the other's verdict before synthesis.
Both receive FixedOutput + PlanSpec only.

Prompts live in config/prompts/critic_a.yaml and critic_b.yaml.
Thinking budgets from routing.yaml.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.validation import (
    CritiqueVerdict, CritiqueRecord, CriticScope, VerdictCategory, ValidationVerdict
)

log = logging.getLogger(__name__)


# ── Critic A ──────────────────────────────────────────────────────────────────

def critic_a_node(state: PipelineState) -> dict:
    """9B non-thinking coherence and completeness critique."""
    run_dir = state["run_dir"]
    fixed   = state.get("fixed_output")
    plan    = state.get("plan_spec")

    if not fixed:
        raise ValueError("critic_a_node: missing fixed_output")

    start = time.perf_counter()

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("critic_a")

    verdict: CritiqueVerdict = call_role(
        role            = "critic_a",
        template_vars   = {
            "plan_json":  plan.model_dump_json(indent=2) if plan else "not available",
            "fixed_json": fixed.model_dump_json(indent=2),
        },
        response_schema = CritiqueVerdict,
        stage           = "critic_a",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(verdict, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(verdict, "_escalated_from", None)
        escalated_models["critic_a"] = escalated_to_attr
        escalation_history.append({
            "stage":      "critic_a",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Critic A escalated %s → %s", escalated_from_attr, escalated_to_attr)

    verdict = verdict.model_copy(update={
        "critic_model": "qwen3.5-9b",
        "scope":        CriticScope.COHERENCE,
        "latency_ms":   (time.perf_counter() - start) * 1000,
    })

    log.info(
        "Critic A (coherence): %s | %d issues | confidence=%s",
        verdict.category, len(verdict.issues), verdict.confidence
    )

    record = _accumulate_verdict(state, verdict)
    return {
        "critique_record":    record,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }


# ── Critic B ──────────────────────────────────────────────────────────────────

def critic_b_node(state: PipelineState) -> dict:
    """DeepCoder 14B thinking correctness verification."""
    run_dir   = state["run_dir"]
    fixed     = state.get("fixed_output")
    plan      = state.get("plan_spec")
    appraisal = state.get("appraisal_report")

    if not fixed:
        raise ValueError("critic_b_node: missing fixed_output")

    start = time.perf_counter()

    appraisal_context = ""
    if appraisal:
        appraisal_context = (
            f"Original AppraisalReport (for reference — evaluate the FixedOutput, "
            f"not the original draft):\n{appraisal.model_dump_json(indent=2)}"
        )

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("critic_b")

    verdict: CritiqueVerdict = call_role(
        role            = "critic_b",
        template_vars   = {
            "plan_json":         plan.model_dump_json(indent=2) if plan else "not available",
            "appraisal_context": appraisal_context,
            "fixed_json":        fixed.model_dump_json(indent=2),
        },
        response_schema = CritiqueVerdict,
        stage           = "critic_b",
        run_dir         = run_dir,
        thinking        = True,
        max_retries     = 0,     # was max_rtreies (typo) — now correct
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(verdict, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(verdict, "_escalated_from", None)
        escalated_models["critic_b"] = escalated_to_attr
        escalation_history.append({
            "stage":      "critic_b",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Critic B escalated %s → %s", escalated_from_attr, escalated_to_attr)

    verdict = verdict.model_copy(update={
        "critic_model": "deepcoder-14b",
        "scope":        CriticScope.CORRECTNESS,
        "latency_ms":   (time.perf_counter() - start) * 1000,
    })

    log.info(
        "Critic B (correctness): %s | %d issues | confidence=%s",
        verdict.category, len(verdict.issues), verdict.confidence
    )

    record = _accumulate_verdict(state, verdict)
    return {
        "critique_record":    record,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _accumulate_verdict(
    state:   PipelineState,
    verdict: CritiqueVerdict,
) -> CritiqueRecord:
    existing = state.get("critique_record")

    if existing:
        updated_verdicts = list(existing.critic_verdicts) + [verdict]
        return existing.model_copy(update={"critic_verdicts": updated_verdicts})

    appraisal = state.get("appraisal_report")
    return CritiqueRecord(
        run_uuid                 = state["run_uuid"],
        iteration                = state.get("iteration", 0),
        task_type                = state.get("task_type", "coding"),
        task_tags                = _derive_tags(state),
        appraisal_critical_count = appraisal.critical_count if appraisal else 0,
        appraisal_major_count    = appraisal.major_count    if appraisal else 0,
        iq2s_inherited_issues    = appraisal.iq2s_inherited_issues if appraisal else [],
        critic_verdicts          = [verdict],
        # Placeholder, overwritten once the synthesis node actually runs
        # (route_after_critic_a -> critic_b -> synthesise -> validate).
        # CritiqueRecord.final_verdict is typed as ValidationVerdict, NOT
        # CritiqueVerdict — a different schema (synthesis_model/description/
        # specific_issues vs. critic_model/issues/confidence). Constructing
        # a CritiqueVerdict here (the previous code) fails Pydantic
        # validation the moment CritiqueRecord(...) is built below, on
        # every single run that reaches critic_a — not just on resume.
        final_verdict            = ValidationVerdict(
            category         = VerdictCategory.PASS,
            synthesis_model  = "pending",
            description      = "pending — awaiting synthesis",
        ),
        resolved         = False,
        total_iterations = state.get("iteration", 0) + 1,
        loops_triggered  = state.get("iteration", 0),
    )


def _derive_tags(state: PipelineState) -> list[str]:
    tags = [state.get("task_type", "coding")]
    plan = state.get("plan_spec")
    if plan and plan.moe_routing_context:
        ctx = plan.moe_routing_context.lower()
        for kw in ["python", "fastapi", "async", "django", "typescript",
                   "react", "database", "rest", "docker", "testing"]:
            if kw in ctx:
                tags.append(kw)
    return list(set(tags))