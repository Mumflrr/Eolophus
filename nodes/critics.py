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
    CritiqueVerdict, CritiqueRecord, CriticScope, VerdictCategory
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
    )

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
    return {"critique_record": record}


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
    )

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
    return {"critique_record": record}


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
        final_verdict            = CritiqueVerdict(
            critic_model = "pending",
            scope        = CriticScope.COHERENCE,
            category     = VerdictCategory.PASS,
            confidence   = "pending",
            reasoning    = "pending",
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
