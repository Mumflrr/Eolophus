"""
Lesson        — a distilled lesson from a CritiqueRecord for LessonL memory.
LessonQuery   — parameters for retrieval scoring.
LessonResult  — a scored lesson returned by the retrieval function.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Lesson(BaseModel):
    """
    Distilled lesson written to SQLite after a successful or instructive run.
    Retrieved and injected into the planning prompt for similar future tasks.
    """
    lesson_uuid:       str       = Field(description="UUID for this lesson record")
    source_run_uuid:   str       = Field(description="Run UUID this lesson was distilled from")

    # Content
    issue_summary:     str       = Field(
        description="One or two sentences describing the issue pattern"
    )
    resolution_pattern:str       = Field(
        description="What fixed it or what the planner should do differently"
    )
    example_context:   Optional[str] = Field(
        default=None,
        description="Brief concrete example from the source run (anonymised if needed)"
    )

    # Tagging for retrieval
    task_type:         str       = Field(description="coding / ideation / mixed")
    tags:              list[str] = Field(
        description=(
            "Searchable tags e.g. ['python', 'async', 'fastapi', 'error_handling']. "
            "Include: language, framework, issue_category, pattern_type."
        )
    )
    model_caught:      str       = Field(
        description="Which model caught this: '9b', 'deepcoder', 'coder14b'"
    )
    issue_category:    str       = Field(
        description="From IssueCategory enum: logic_error, spec_delta, etc."
    )

    # Scoring metadata
    confidence_score:  float     = Field(
        default=1.0,
        description="Incremented each time this lesson is reinforced. Starts at 1.0."
    )
    times_seen:        int       = Field(default=1)
    times_retrieved:   int       = Field(default=0)
    times_useful:      int       = Field(
        default=0,
        description="Incremented when retrieved lesson appears in 9B reasoning output"
    )
    last_triggered:    Optional[str] = Field(
        default=None,
        description="ISO timestamp of last time this lesson was triggered"
    )

    # Meta-distillation
    is_meta_lesson:    bool      = Field(
        default=False,
        description="True if this was produced by periodic consolidation of multiple lessons"
    )
    source_lesson_uuids: list[str] = Field(
        default_factory=list,
        description="If meta-lesson: UUIDs of constituent lessons"
    )


class LessonQuery(BaseModel):
    """Parameters for lesson retrieval scoring."""
    task_type:       str       = Field(description="Must match exactly")
    tags:            list[str] = Field(description="Current task tags to match against")
    top_k:           int       = Field(default=5)
    min_score:       float     = Field(
        default=0.2,
        description="Minimum score threshold — lessons below this are not injected"
    )


class LessonResult(BaseModel):
    """A scored lesson returned by the retrieval function."""
    lesson:          Lesson = Field()
    score:           float  = Field(description="Composite relevance score 0.0–1.0")
    score_breakdown: dict   = Field(
        default_factory=dict,
        description="Component scores: confidence, recency, similarity, difficulty"
    )
