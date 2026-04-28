"""
PipelineState — the single state object passed between all LangGraph nodes.
Every node receives this, does its work, and returns a dict of only the
fields it modified. LangGraph merges updates automatically.

Artefacts are written to disk as they are produced. State carries file paths,
not content, except for small classification and routing objects.
"""

from __future__ import annotations
from typing import Optional, TypedDict

from schemas.task_classification import TaskClassification, Mode, TaskType
from schemas.visual_description   import VisualDescription
from schemas.ideation_output      import IdeationOutput
from schemas.plan_spec            import PlanSpec
from schemas.execution            import DraftOutput, AppraisalReport, FixedOutput
from schemas.validation           import CritiqueRecord, ValidationVerdict
from schemas.sub_spec             import SubSpecInterface


class PipelineState(TypedDict, total=False):
    """
    Central state object. Fields are optional (total=False) because nodes
    only populate fields relevant to their stage.

    Naming convention:
      *_path  — path to a JSON artefact on disk
      *_obj   — in-memory Pydantic object (small, not written to disk separately)
    """

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_uuid:           str             # UUID for this run; also the runs/ directory name
    run_dir:            str             # Absolute path to runs/{run_uuid}/
    mode:               str             # "short" | "long"  (from TaskClassification)
    task_type:          str             # "coding" | "ideation" | "mixed"
    is_sub_spec:        bool            # True if this is a sub-spec run
    parent_run_uuid:    Optional[str]   # Set if is_sub_spec is True
    iteration:          int             # Correction loop iteration count (0-indexed)

    # ── Classification ────────────────────────────────────────────────────────
    classification:     Optional[TaskClassification]

    # ── Raw input ─────────────────────────────────────────────────────────────
    raw_text_input:     Optional[str]
    raw_image_path:     Optional[str]   # Path to uploaded image if visual input

    # ── Vision stage ──────────────────────────────────────────────────────────
    visual_description: Optional[VisualDescription]
    normalised_input:   Optional[str]   # Text task description after vision normalisation

    # ── Ideation stage ────────────────────────────────────────────────────────
    ideation_output:    Optional[IdeationOutput]
    ideation_path:      Optional[str]   # Path to ideation.json on disk

    # ── Planning stage ────────────────────────────────────────────────────────
    plan_spec:          Optional[PlanSpec]
    plan_spec_path:     Optional[str]   # Path to planspec.json — persists entire run
    relevant_lessons:   Optional[list]  # LessonResult list injected into planning prompt

    # ── Draft stage ───────────────────────────────────────────────────────────
    draft_output:       Optional[DraftOutput]
    draft_path:         Optional[str]   # Path to draft.json — overwritten on redraft

    # ── Appraisal stage ───────────────────────────────────────────────────────
    appraisal_report:   Optional[AppraisalReport]
    appraisal_path:     Optional[str]   # Path to appraisal_report.json

    # ── Bug fix stage ─────────────────────────────────────────────────────────
    fixed_output:       Optional[FixedOutput]
    fixed_path:         Optional[str]   # Path to fixed.json

    # ── Critique ensemble ─────────────────────────────────────────────────────
    critique_record:    Optional[CritiqueRecord]
    critique_path:      Optional[str]   # Path to critique.json

    # ── Validation ────────────────────────────────────────────────────────────
    validation_verdict: Optional[ValidationVerdict]
    verdict_path:       Optional[str]   # Path to verdict.json

    # ── Sub-spec decomposition ────────────────────────────────────────────────
    decompose:          bool
    sub_spec_uuids:     Optional[list[str]]     # UUIDs of sub-spec runs if decomposed
    sub_spec_interfaces:Optional[list[SubSpecInterface]]
    final_validation_path: Optional[str]        # Path to final_validation.json

    # ── Final output ──────────────────────────────────────────────────────────
    final_output_path:  Optional[str]   # Path to final.json
    pipeline_complete:  bool
    pipeline_failed:    bool
    failure_reason:     Optional[str]
