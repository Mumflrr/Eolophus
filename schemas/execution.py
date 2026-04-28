"""
DraftOutput    — produced by the 35B MoE draft generation stage.
AppraisalReport— produced by DeepCoder 14B correctness appraisal stage.
FixedOutput    — produced by Qwen2.5 Coder 14B bug fix stage.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from schemas.confidence import ConfidenceMixin


# ── DraftOutput ───────────────────────────────────────────────────────────────

class ComponentDraft(BaseModel):
    component_name: str = Field(description="Must match a ComponentSpec.name from the PlanSpec")
    code:           str = Field(description="Complete implementation code for this component")
    notes:          Optional[str] = Field(
        default=None,
        description="Any decisions made during implementation that deviate from the spec"
    )


class DraftOutput(ConfidenceMixin, BaseModel):
    """
    Complete draft produced by the 35B MoE.
    Passed to DeepCoder for correctness appraisal alongside the original PlanSpec.
    """
    component_drafts: list[ComponentDraft] = Field(
        description="One entry per component in PlanSpec.implementation_order"
    )
    implementation_notes: Optional[str] = Field(
        default=None,
        description="Overall notes about architectural decisions made during drafting"
    )
    deviations_from_spec: list[str] = Field(
        default_factory=list,
        description=(
            "Any places where the implementation consciously deviated from the PlanSpec "
            "and why. Empty if fully spec-compliant."
        )
    )
    # The confidence_signal field was removed; ConfidenceMixin now handles this cleanly.


# ── AppraisalReport ───────────────────────────────────────────────────────────

class IssueSeverity(str, Enum):
    CRITICAL = "critical"   # Will cause failure or incorrect behaviour
    MAJOR    = "major"      # Significant quality or correctness problem
    MINOR    = "minor"      # Style, naming, or minor convention issue


class IssueCategory(str, Enum):
    LOGIC_ERROR         = "logic_error"
    CONSTRAINT_VIOLATION= "constraint_violation"
    SPEC_DELTA          = "spec_delta"          # Implementation differs from spec
    MISSING_REQUIREMENT = "missing_requirement"
    ERROR_HANDLING      = "error_handling"
    TYPE_ERROR          = "type_error"
    EDGE_CASE           = "edge_case"
    PERFORMANCE         = "performance"
    OTHER               = "other"


class Issue(BaseModel):
    component:   str           = Field(description="Component name where issue was found")
    severity:    IssueSeverity = Field()
    category:    IssueCategory = Field()
    description: str           = Field(description="Clear description of the issue")
    location:    Optional[str] = Field(
        default=None,
        description="Function name or line reference if applicable"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Direction for fixing — not replacement code, just guidance"
    )


class AppraisalReport(ConfidenceMixin, BaseModel):
    """
    Correctness appraisal produced by DeepCoder 14B.
    Received by Coder 14B alongside the DraftOutput.
    Also stored in CritiqueRecord for lineage tracking.
    """
    overall_assessment: str = Field(
        description=(
            "One paragraph summary: does the draft satisfy the PlanSpec? "
            "What is the most significant concern if any?"
        )
    )
    spec_satisfaction: str = Field(
        description="high / partial / low — how well does the draft satisfy the PlanSpec?"
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="All identified issues, ordered by severity descending"
    )
    critical_count: int = Field(description="Number of critical severity issues")
    major_count:    int = Field(description="Number of major severity issues")
    minor_count:    int = Field(description="Number of minor severity issues")
    components_reviewed: list[str] = Field(
        description="Names of components reviewed — should match all in DraftOutput"
    )
    iq2s_inherited_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Issues likely inherited from IQ2_S ideation imprecision "
            "rather than execution errors. Used for lineage tracking."
        )
    )


# ── FixedOutput ───────────────────────────────────────────────────────────────

class AppliedFix(BaseModel):
    issue_description: str = Field(description="Brief description of the issue that was fixed")
    component:         str = Field(description="Component where fix was applied")
    fix_description:   str = Field(description="What was changed and how")


class FixedOutput(BaseModel):
    """
    Bug-fixed and idiomatically reviewed output from Coder 14B.
    This is what the critique ensemble and final validation receive.
    """
    component_drafts: list[ComponentDraft] = Field(
        description="Fixed implementation — same structure as DraftOutput.component_drafts"
    )
    applied_fixes: list[AppliedFix] = Field(
        default_factory=list,
        description="Record of all fixes applied, including both appraisal-directed and self-identified"
    )
    self_identified_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Issues Coder 14B identified independently during its own review "
            "that were not in the AppraisalReport"
        )
    )
    unfixed_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Issues from the AppraisalReport that could not be fixed at this stage "
            "and may require re-planning"
        )
    )
    overall_quality: str = Field(
        description="Brief assessment of output quality after fixes"
    )