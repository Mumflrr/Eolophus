"""
Schemas package — all Pydantic contracts for the pipeline.
Import from here rather than from individual modules.
"""

from schemas.task_classification import (
    TaskClassification, Mode, TaskType, Complexity
)
from schemas.visual_description import (
    VisualDescription, UIElement
)
from schemas.ideation_output import (
    IdeationOutput, Approach
)
from schemas.plan_spec import (
    PlanSpec, ComponentSpec, FunctionSpec, Parameter, DroppedIdea
)
from schemas.execution import (
    DraftOutput, ComponentDraft,
    AppraisalReport, Issue, IssueSeverity, IssueCategory,
    FixedOutput, AppliedFix
)
from schemas.validation import (
    CritiqueVerdict, ValidationVerdict, CritiqueRecord,
    VerdictCategory, CriticScope
)
from schemas.sub_spec import (
    SubSpecInterface, SharedObjectRef, InterfaceStatus
)
from schemas.lesson import (
    Lesson, LessonQuery, LessonResult
)

__all__ = [
    # Classification
    "TaskClassification", "Mode", "TaskType", "Complexity",
    # Vision
    "VisualDescription", "UIElement",
    # Ideation
    "IdeationOutput", "Approach",
    # Planning
    "PlanSpec", "ComponentSpec", "FunctionSpec", "Parameter", "DroppedIdea",
    # Execution
    "DraftOutput", "ComponentDraft",
    "AppraisalReport", "Issue", "IssueSeverity", "IssueCategory",
    "FixedOutput", "AppliedFix",
    # Validation
    "CritiqueVerdict", "ValidationVerdict", "CritiqueRecord",
    "VerdictCategory", "CriticScope",
    # Sub-spec
    "SubSpecInterface", "SharedObjectRef", "InterfaceStatus",
    # Lessons
    "Lesson", "LessonQuery", "LessonResult",
]
