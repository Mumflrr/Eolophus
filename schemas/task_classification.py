"""
TaskClassification — produced by the 9B at the start of every run.
Determines mode, task type, complexity, and whether to decompose into sub-specs.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


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

    mode/task_type/complexity are Optional here — previously required,
    non-nullable enums. On a genuinely ambiguous task (e.g. "Please do
    the task"), the model correctly recognises it can't classify at all
    and tries to say so — but a required enum has no legal way to
    represent "unknown", only a value from the enum's literal set. The
    result was a guaranteed InstructorRetryException on first attempt:
    Pydantic validation error type=enum, input_value=None for all three
    fields simultaneously, even though the model's own reasoning_content
    (and its populated confidence="low" + clarification_question) shows
    it was trying to do exactly the right thing. classify_node's outer
    retry loop then recovers on attempt 2 — but only by feeding the
    validation error back and getting the model to force SOME value in
    rather than surfacing the clarification question it originally
    wanted to ask, which is the wrong outcome for a task that really is
    unclassifiable without more information from the user.
    A model_validator below closes the gap the other direction: null is
    only accepted when confidence="low", so a confident classification
    still can't slip a None past validation. Together, this lets the
    model actually express "I don't know" the way classify.yaml's own
    prompt already asks it to, instead of being forced into a guess.
    """
    mode: Optional[Mode] = Field(
        default=None,
        description=(
            "short = interactive, 9B executes, no ideation. "
            "long  = batch, 35B executes, ideation fires if open-ended. "
            "Set to null ONLY when confidence is low and the task is too "
            "ambiguous to classify at all — populate clarification_question "
            "in that case. Must be set to a real value whenever confidence "
            "is 'high' or 'medium'."
        )
    )
    task_type: Optional[TaskType] = Field(
        default=None,
        description=(
            "Primary nature of the task. Set to null ONLY when confidence "
            "is low and the task is too ambiguous to classify at all — "
            "populate clarification_question in that case. Must be set to "
            "a real value whenever confidence is 'high' or 'medium'."
        )
    )
    complexity: Optional[Complexity] = Field(
        default=None,
        description=(
            "simple   = single function / component, clear spec. "
            "moderate = multi-component, some ambiguity. "
            "complex  = architectural, cross-cutting, or multi-file. "
            "Set to null ONLY when confidence is low and the task is too "
            "ambiguous to classify at all — populate clarification_question "
            "in that case. Must be set to a real value whenever confidence "
            "is 'high' or 'medium'."
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

    # Added alongside the mode/task_type/complexity -> Optional change
    # above. Making those three fields nullable closes the "model has no
    # legal way to express unknown" gap, but on its own it would also
    # silently allow a CONFIDENT classification (confidence="high" or
    # "medium") to leave them null — which defeats the whole point of
    # classification and would push a None straight into
    # pipeline/routers.py's select_profile()/routing logic, which reads
    # classification.complexity/decompose unconditionally and has no null
    # handling of its own. This validator restores that guarantee: null
    # is accepted ONLY when confidence="low", exactly mirroring
    # classify.yaml's own stated rule ("low = genuinely ambiguous... 
    # Populate clarification_question"). Every other case is held to the
    # original required-value behaviour.
    @model_validator(mode="after")
    def _require_classification_unless_low_confidence(self) -> "TaskClassification":
        if self.confidence != "low":
            missing = [
                name for name in ("mode", "task_type", "complexity")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"mode/task_type/complexity must be set to a real value "
                    f"when confidence is '{self.confidence}' (missing: "
                    f"{missing}). Only confidence='low' may leave these "
                    f"null — set confidence='low' and populate "
                    f"clarification_question instead, or provide a real "
                    f"value for every missing field."
                )
        return self