"""
TaskClassification — produced by the 9B at the start of every run.
Determines mode, task type, complexity, and whether to decompose into sub-specs.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Mode(str, Enum):
    SHORT = "short"
    LONG  = "long"


class TaskType(str, Enum):
    CODING   = "coding"
    IDEATION = "ideation"
    MIXED    = "mixed"


class Complexity(str, Enum):
    SIMPLE   = "simple"
    MODERATE = "moderate"
    COMPLEX  = "complex"


class TaskClassification(BaseModel):
    """
    9B classification output produced before any planning begins.
    All downstream routing reads from this object.
    """
    mode: Mode = Field(
        description=(
            "short = interactive, 9B executes, no ideation. "
            "long  = batch, 35B executes, ideation fires if open-ended."
        )
    )
    task_type: TaskType = Field(
        description="Primary nature of the task."
    )
    complexity: Complexity = Field(
        description=(
            "simple   = single function / component, clear spec. "
            "moderate = multi-component, some ambiguity. "
            "complex  = architectural, cross-cutting, or multi-file."
        )
    )
    decompose: bool = Field(
        description=(
            "True if the task should be split into sub-specs. "
            "Decompose when: more than 5 independent components, "
            "or any single component description would exceed 500 tokens."
        )
    )
    estimated_sub_specs: Optional[int] = Field(
        default=None,
        description="Estimated number of sub-specs if decompose is True. Omit otherwise."
    )
    reasoning: str = Field(
        description=(
            "Brief explanation of the classification decisions. "
            "Used for debugging misclassifications."
        )
    )

    model_config = {"use_enum_values": True}
