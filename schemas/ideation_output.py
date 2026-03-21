"""
IdeationOutput — produced by the 27B IQ2_S ideation stage (long mode only).
Broad exploration of the problem space. Precision imprecision is expected
and explicitly filtered by the 9B consistency check before planning.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Approach(BaseModel):
    title:       str  = Field(description="Short name for this approach")
    description: str  = Field(description="What this approach involves and why it might suit the task")
    tradeoffs:   str  = Field(description="Key tradeoffs — what it gains, what it costs")
    feasibility: str  = Field(description="rough assessment: high / medium / speculative")


class IdeationOutput(BaseModel):
    """
    27B IQ2_S exploration of the problem space.
    NOT a plan. Ideas here will be filtered by the 9B before any planning occurs.
    Downstream models do not receive this object directly.
    """
    problem_restatement: str = Field(
        description="The task as the model understood it — used to catch misinterpretations."
    )
    approaches: list[Approach] = Field(
        description="Two to five distinct approaches to the problem. Breadth over precision."
    )
    architectural_directions: list[str] = Field(
        default_factory=list,
        description="High-level architectural patterns or technology choices worth considering."
    )
    potential_components: list[str] = Field(
        default_factory=list,
        description="Likely components, modules, or layers the solution might involve."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions whose answers would meaningfully change the approach."
    )
    recommended_direction: Optional[str] = Field(
        default=None,
        description=(
            "If one approach seems clearly preferable, name it and briefly say why. "
            "Optional — omit if genuinely unclear."
        )
    )
