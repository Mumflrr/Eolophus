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
    DESCRIBE = "describe"


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
    # Added: previously only existed on nodes/classifier.py's own duplicate
    # TaskClassification class, which classify_node actually built and
    # returned instead of this one. LangGraph doesn't enforce PipelineState's
    # TypedDict annotations at runtime, so nothing caught the mismatch on a
    # fresh run — but the checkpoint serializer bakes in the concrete class's
    # module path, so a resume deserializes 'nodes.classifier.TaskClassification'
    # and logs "Deserializing unregistered type ... This will be blocked in a
    # future version." Consolidating on one class here (and having
    # nodes/classifier.py import it instead of redefining it) fixes that,
    # and means clarify's confidence/clarification_question fields and the
    # "describe" task_type (both load-bearing for existing routing in
    # pipeline/routers.py) are actually declared where PipelineState expects
    # TaskClassification to come from, instead of only existing on a class
    # nothing outside classify_node ever imports.
    confidence: str = Field(
        default="high",
        description="high=proceed. medium=proceed with warning. low=halt and clarify."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Single specific question to resolve ambiguity. Only when confidence=low."
    )

    model_config = {"use_enum_values": True}