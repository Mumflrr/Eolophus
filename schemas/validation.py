"""
CritiqueVerdict  — produced by each critic in the ensemble.
ValidationVerdict— produced by the synthesis model; drives routing.
CritiqueRecord   — assembled record of a full ensemble run for storage.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Shared enums ──────────────────────────────────────────────────────────────

class VerdictCategory(str, Enum):
    PASS            = "pass"
    MINOR_FIX       = "minor_fix"
    SPEC_PROBLEM    = "spec_problem"
    UNRESOLVABLE    = "unresolvable"


class CriticScope(str, Enum):
    COHERENCE    = "coherence"      # Critic A — 9B non-thinking
    CORRECTNESS  = "correctness"    # Critic B — DeepCoder thinking


# ── CritiqueVerdict ───────────────────────────────────────────────────────────

class CritiqueVerdict(BaseModel):
    """
    Independent verdict from one critic.
    Critics do not see each other's verdicts before synthesis.
    """
    critic_model:  str         = Field(description="Model identifier e.g. 'qwen3.5-9b', 'deepcoder-14b'")
    scope:         CriticScope = Field(description="What dimension this critic evaluated")
    category:      VerdictCategory = Field(description="High-level verdict")
    issues:        list[str]   = Field(
        default_factory=list,
        description="Specific issues found. Empty if category is pass."
    )
    confidence:    str         = Field(description="high / medium / low")
    reasoning:     str         = Field(
        description="Brief reasoning for the verdict — used by synthesis and stored for calibration"
    )
    tokens_used:   Optional[int]   = Field(default=None)
    latency_ms:    Optional[float] = Field(default=None)

    model_config = {"use_enum_values": True}


# ── ValidationVerdict ─────────────────────────────────────────────────────────

class ValidationVerdict(BaseModel):
    """
    Final consolidated verdict from the synthesis model.
    This is what LangGraph's routing edge switches on.
    """
    category: VerdictCategory = Field(
        description=(
            "pass          = output satisfies PlanSpec; exit loop. "
            "minor_fix     = specific issues; route to executor for targeted redraft. "
            "spec_problem  = plan is flawed; route to planner. "
            "unresolvable  = iteration limit hit; return best attempt."
        )
    )
    synthesis_model: str = Field(description="Which model produced this synthesis")
    description:     str = Field(description="Summary of the verdict and key findings")
    specific_issues: list[str] = Field(
        default_factory=list,
        description="Actionable issue descriptions for the executor or planner. Empty if pass."
    )
    dissenting_notes: Optional[str] = Field(
        default=None,
        description=(
            "Cases where synthesis overrode a critic's concern. "
            "Critical for retrospective calibration."
        )
    )
    suggested_fix_direction: Optional[str] = Field(
        default=None,
        description="High-level direction for the fix if category is minor_fix"
    )
    tokens_used:  Optional[int]   = Field(default=None)
    latency_ms:   Optional[float] = Field(default=None)

    model_config = {"use_enum_values": True}


# ── CritiqueRecord ────────────────────────────────────────────────────────────

class CritiqueRecord(BaseModel):
    """
    Full record of one ensemble run. Written to Langfuse and SQLite.
    Carries lineage information for IQ2_S tracking.
    """
    run_uuid:         str = Field(description="Parent run UUID")
    iteration:        int = Field(description="Correction loop iteration number (0-indexed)")
    task_type:        str = Field(description="From TaskClassification")
    task_tags:        list[str] = Field(default_factory=list, description="For lesson retrieval scoring")

    # Appraisal lineage
    appraisal_critical_count: int = Field(default=0)
    appraisal_major_count:    int = Field(default=0)
    iq2s_inherited_issues:    list[str] = Field(
        default_factory=list,
        description="Issues flagged as IQ2_S-originated in the AppraisalReport"
    )

    # Critic verdicts
    critic_verdicts: list[CritiqueVerdict] = Field(default_factory=list)

    # Synthesis
    final_verdict: ValidationVerdict = Field()

    # Outcome
    resolved:          bool = Field(description="Did this run ultimately pass validation?")
    total_iterations:  int  = Field(description="Total correction iterations for this run")
    loops_triggered:   int  = Field(description="Correction loops triggered in this run")

    # Timing
    ensemble_latency_ms: Optional[float] = Field(default=None)
