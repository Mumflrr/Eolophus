"""
nodes/critics.py — critique ensemble.

Critic A: 9B non-thinking — coherence and completeness.
Critic B: DeepCoder 14B thinking — logical correctness verification.

Critics run independently. Neither sees the other's verdict before synthesis.
Both receive FixedOutput + PlanSpec only.

CritiqueVerdicts are accumulated into the CritiqueRecord in state.
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

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_CRITIC_A = """You are Critic A: coherence and completeness reviewer.

Evaluate the FixedOutput against the PlanSpec from these angles ONLY:
  - Does the output hang together end-to-end? No contradictions between components?
  - Are all components from the PlanSpec present and non-empty?
  - Are all edge cases the PlanSpec specified addressed somewhere?
  - Does the overall structure make sense for the stated goal?
  - Are there obvious missing pieces that would prevent the output from working?

You are NOT evaluating code correctness or implementation details.
That is another critic's job. Stay within your scope.

Return a CritiqueVerdict with category:
  pass          — output is coherent and complete
  minor_fix     — small gaps or inconsistencies that can be fixed without replanning
  spec_problem  — fundamental structural issue requiring replanning
"""

_SYSTEM_CRITIC_B = """You are Critic B: logical correctness verifier.

You have previously appraised a draft and seen fixes applied.
Now verify whether the FixedOutput actually resolves the concerns and is logically correct.

Evaluate from these angles ONLY:
  - Does the implementation satisfy the logical requirements of the PlanSpec?
  - Are there constraint violations — things the spec required that are violated?
  - Are there spec deltas — subtle differences between what was specified and what was implemented?
  - Are there logic errors that would cause incorrect behaviour at runtime?
  - Are there unhandled edge cases the spec identified?

You are NOT evaluating coding style, idioms, or conventions.
That is outside your scope here.

When your reasoning is complete, emit: <confidence>high</confidence>

Return a CritiqueVerdict with category:
  pass          — implementation is logically correct and satisfies the spec
  minor_fix     — fixable logic issues that don't require replanning
  spec_problem  — fundamental logical flaw requiring a new plan
"""


# ── Critic A ──────────────────────────────────────────────────────────────────

def critic_a_node(state: PipelineState) -> dict:
    """
    9B non-thinking coherence and completeness critique.
    Always runs when ensemble is active.
    """
    run_dir = state["run_dir"]
    fixed   = state.get("fixed_output")
    plan    = state.get("plan_spec")

    if not fixed:
        raise ValueError("critic_a_node: missing fixed_output")

    start = time.perf_counter()

    messages = [
        {"role": "system", "content": _SYSTEM_CRITIC_A},
        {
            "role": "user",
            "content": (
                f"PlanSpec:\n{plan.model_dump_json(indent=2) if plan else 'not available'}\n\n"
                f"FixedOutput to review:\n{fixed.model_dump_json(indent=2)}"
            ),
        },
    ]

    verdict: CritiqueVerdict = call_role(
        role            = "critic_a",
        messages        = messages,
        response_schema = CritiqueVerdict,
        stage           = "critic_a",
        run_dir         = run_dir,
        thinking        = False,
    )

    # Ensure scope is set correctly
    verdict = verdict.model_copy(update={
        "critic_model": "qwen3.5-9b",
        "scope":        CriticScope.COHERENCE,
        "latency_ms":   (time.perf_counter() - start) * 1000,
    })

    log.info(
        "Critic A (coherence): %s | %d issues | confidence=%s",
        verdict.category, len(verdict.issues), verdict.confidence
    )

    # Accumulate into existing critique_record or start new one
    record = _accumulate_verdict(state, verdict)

    return {"critique_record": record}


# ── Critic B ──────────────────────────────────────────────────────────────────

def critic_b_node(state: PipelineState) -> dict:
    """
    DeepCoder 14B thinking correctness verification.
    Optional — runs when ensemble config enables it.
    """
    run_dir = state["run_dir"]
    fixed   = state.get("fixed_output")
    plan    = state.get("plan_spec")
    appraisal = state.get("appraisal_report")

    if not fixed:
        raise ValueError("critic_b_node: missing fixed_output")

    start = time.perf_counter()

    # Build context: include appraisal for reference but instruct critic
    # to evaluate the FIXED output, not the original draft
    appraisal_ctx = ""
    if appraisal:
        appraisal_ctx = (
            f"\n\nOriginal AppraisalReport (for reference — evaluate the FixedOutput, "
            f"not the original draft):\n{appraisal.model_dump_json(indent=2)}"
        )

    messages = [
        {"role": "system", "content": _SYSTEM_CRITIC_B},
        {
            "role": "user",
            "content": (
                f"PlanSpec:\n{plan.model_dump_json(indent=2) if plan else 'not available'}"
                f"{appraisal_ctx}\n\n"
                f"FixedOutput to verify:\n{fixed.model_dump_json(indent=2)}"
            ),
        },
    ]

    verdict: CritiqueVerdict = call_role(
        role            = "critic_b",
        messages        = messages,
        response_schema = CritiqueVerdict,
        stage           = "critic_b",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = 4096,
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
    """
    Add a CritiqueVerdict to the existing CritiqueRecord, or create one.
    The record is finalised (with ValidationVerdict) by the synthesiser.
    """
    existing = state.get("critique_record")
    classification = state.get("classification")

    if existing:
        # Append to existing verdicts list
        updated_verdicts = list(existing.critic_verdicts) + [verdict]
        return existing.model_copy(update={"critic_verdicts": updated_verdicts})

    # Create new record
    appraisal = state.get("appraisal_report")
    return CritiqueRecord(
        run_uuid                  = state["run_uuid"],
        iteration                 = state.get("iteration", 0),
        task_type                 = state.get("task_type", "coding"),
        task_tags                 = _derive_tags(state),
        appraisal_critical_count  = appraisal.critical_count if appraisal else 0,
        appraisal_major_count     = appraisal.major_count    if appraisal else 0,
        iq2s_inherited_issues     = appraisal.iq2s_inherited_issues if appraisal else [],
        critic_verdicts           = [verdict],
        # final_verdict filled by synthesiser
        final_verdict             = CritiqueVerdict(
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
