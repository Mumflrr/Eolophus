"""
schemas/confidence.py — confidence and clarification fields.

Added to TaskClassification and PlanSpec to enable:
  1. Halting the pipeline when the 9B is uncertain
  2. Surfacing a specific clarification question to the caller
  3. Logging confidence as a quality signal per stage

Confidence levels:
  high   — proceed normally
  medium — proceed with a warning in stages.log
  low    — halt and return clarification_question to caller
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceMixin(BaseModel):
    """
    Mixin added to schemas where uncertainty should halt the pipeline.
    Include these fields in TaskClassification and PlanSpec.
    """
    confidence: str = Field(
        default="high",
        description=(
            "Model's confidence in this output. "
            "high   = proceed normally. "
            "medium = proceed but log warning. "
            "low    = halt: populate clarification_question."
        )
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "If confidence is low, a single specific question whose answer "
            "would resolve the uncertainty. "
            "Examples: "
            "'Should the rate limiter persist state across restarts?' "
            "'Is this a REST API or a CLI tool?' "
            "Leave null for medium or high confidence."
        )
    )
