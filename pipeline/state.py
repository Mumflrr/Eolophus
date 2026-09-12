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
    # Set from RunRequest.use_search (server.py) / ChatMessageIn's replan
    # path. Was previously passed into initial_state / turn_state without
    # being declared here, so LangGraph silently dropped it before any
    # node could read it — see the custom_step_outputs comment below for
    # why undeclared top-level keys don't survive. plan_node/ideation_node
    # should check this and call clients.search.search_web() when true.
    use_search:         bool

    # ── Pipeline profile & escalation ─────────────────────────────────────────
    # profile: the resolved pipeline_profiles (routing.yaml) name for this
    # run — "short" | "medium" | "long" | "ultra". Set once at classify_node
    # (or immediately after, for "auto" — see select_profile in
    # pipeline/routers.py) and never changed for the rest of the run; this
    # is what routers.py now consults instead of classification.mode, which
    # is informational only going forward. Distinct from
    # requested_profile below because "auto" needs to record both what the
    # caller asked for and what it actually resolved to (useful for the
    # run-detail UI and for debugging auto-selection).
    profile:             Optional[str]
    requested_profile:   Optional[str]   # "auto" | "short" | "medium" | "long" | "ultra" — what the caller asked for, before auto-resolution
    human_in_the_loop:   bool            # False = "set-and-forget": low confidence escalates instead of halting, and proceeds best-effort once the ladder is exhausted rather than waiting on /clarify
    # Per-stage current-model overrides produced by escalation, keyed by
    # stage name (e.g. {"draft": "35b"}). A stage with no entry here is
    # still on whatever models.yaml roles:/the active profile's
    # role_overrides says. clients/llm.py's call_role() checks this dict
    # BEFORE falling back to roles:/role_overrides, and updates it (via the
    # node's returned state dict) every time a stage actually escalates.
    # Declared as a plain dict (not a per-stage top-level key) per the
    # custom_step_outputs rule below — a dynamically-named key per stage
    # would be silently dropped by LangGraph.
    escalated_models:    dict
    # Human-readable escalation history for this run, appended to (never
    # overwritten) each time any stage escalates. Consumed by distiller.py
    # to make sure the substantive lesson written is about the task, not
    # the escalation event itself (design doc §2.3) — this list is where
    # the "it escalated" fact lives instead, kept separate from whatever
    # distiller.py writes to the lesson store. Each entry:
    # {"stage": str, "from_model": str, "to_model": str, "trigger": "truncation"|"low_confidence", "iteration": int}
    escalation_history:  list

    # ── Classification ────────────────────────────────────────────────────────
    classification:     Optional[TaskClassification]

    # ── Raw input ─────────────────────────────────────────────────────────────
    raw_text_input:     Optional[str]
    raw_image_path:     Optional[str]   # Path to uploaded image if visual input
    # Live (non-excluded) attachments as plain {filename, content} dicts —
    # same shape _load_live_attachments/_compose_input_with_attachments use
    # in server.py. Declared here per the custom_step_outputs comment below:
    # LangGraph only tracks top-level keys declared on this TypedDict, so an
    # undeclared "attachments" key risked being silently dropped. The main
    # pipeline doesn't strictly depend on this key surviving — every node
    # reads attachment content indirectly via normalised_input/raw_text_input,
    # which server.py composes before invoke() — but distiller.py's
    # _scrub_attachment_references reads state.get("attachments") directly,
    # and sub_spec_runner_node (pipeline/graph.py) needs it to fold
    # attachment content into each sub-spec's own task_input.
    attachments:        Optional[list[dict]]

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
    pipeline_halted: bool
    clarification_needed: Optional[str]
    # Set by classify_node when TaskClassification.confidence == "low".
    # Must be declared here (not just returned from the node) or LangGraph
    # drops it before clarify_node can read it back out of committed state —
    # see the long comment in clarify_node (pipeline/graph.py) for how this
    # was diagnosed.
    clarification_question: Optional[str]

    # ── Custom pipeline bookkeeping ──────────────────────────────────────
    # CRITICAL: LangGraph only tracks top-level keys declared on this
    # TypedDict — any key a node returns that ISN'T listed here is
    # silently dropped (confirmed empirically, not a guess). Per-step
    # dynamic data (one decision node's outcome, one freeform node's
    # output) therefore CANNOT use a dynamically-named top-level key like
    # f"_decision__{step_id}" — it must live INSIDE one of these fixed
    # dict-valued fields instead, since dict contents aren't subject to
    # this restriction, only the TypedDict's own key set is.
    custom_step_outputs: dict       # step_id -> whatever that step produced
    custom_decisions:    dict       # decision_step_id -> chosen outcome value
    custom_iter_counts:  dict       # decision_step_id -> times visited
    custom_feedback:     Optional[str]  # most recent decision's reasoning, or None
    global_step_count:   int