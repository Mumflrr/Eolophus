"""
api/schemas.py — wire-format pydantic models for the Eolophus API.

All request/response bodies live here so routers can import just the
shapes they need without pulling in unrelated endpoint logic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AttachmentIn(BaseModel):
    """A text-based file attached to a run (code, markdown, plain text, etc.).
    Not for images — see ImageIn / RunRequest.image below."""
    filename: str
    content:  str   # raw text content, already decoded client-side


class ImageIn(BaseModel):
    """
    A single image attached to a run, routed through vision_decode_node
    (nodes/vision.py) rather than the text attachment path above.

    One image per run, matching vision_decode_node's state shape
    (raw_image_path is a single path, not a list) and route_after_input's
    binary vision-vs-classify branch in pipeline/graph.py. If multi-image
    support is wanted later, both that router and vision_decode_node's
    single-image base64 encode would need to change together — this isn't
    just an API-layer limit.

    data_url is the full "data:image/png;base64,...." string as produced
    by the browser's FileReader.readAsDataURL — kept as one string rather
    than splitting mime type out of the client, since the server needs to
    parse it into raw bytes anyway to decide the file extension it saves
    under (see attachments.write_image).
    """
    filename: str
    data_url: str


class AddAttachmentsIn(BaseModel):
    """Body for POST /run/{run_uuid}/attachments — adding files to a run
    already in progress, from the chat view rather than at creation time."""
    attachments: list[AttachmentIn]


class RunRequest(BaseModel):
    task:               str
    # "" / "auto" / None = let classify_node's select_profile() decide;
    # else one of "short" | "medium" | "long" | "ultra" — pins the
    # resolved pipeline_profiles entry directly. Replaces the old
    # mode + PIPELINE_ULTRA/PIPELINE_FORCE_SHORT env-flag mechanism (see
    # docs/pipeline-profile-escalation-design.md); mode is now purely an
    # informational field on TaskClassification and no longer drives
    # routing here.
    requested_profile: Optional[str] = None
    task_type:          Optional[str] = None   # "coding" | "ideation" | "mixed" | "describe"
    # Gates whether classify/plan halt to ask before escalating to a
    # bigger model (True, default) or auto-escalate/proceed best-effort
    # with no one to ask (False, "set-and-forget") — see classifier.py's
    # EscalationNeeded handling for the full ask-then-decide flow.
    human_in_the_loop: bool = True
    use_search:         bool = False
    attachments:        list[AttachmentIn] = []   # text/code files attached to this run
    image:              Optional[ImageIn]  = None  # optional image -> vision_decode_node
    pipeline:           Optional[str] = None   # None = built-in pipeline; else a name
                                            # from config/pipelines/{name}.yaml,
                                            # created via the pipeline CRUD endpoints.
                                            # requested_profile/task_type are ignored when
                                            # pipeline is set — custom pipelines define
                                            # their own model/budget choices per step.


class PipelineStepIn(BaseModel):
    """Wire-format mirror of schemas.pipeline_def's step union, for the CRUD API."""
    type: str    # "existing" | "freeform" | "decision"
    id:   str

    # existing
    node_name:      Optional[str] = None
    model_override: Optional[str] = None
    budget_override:Optional[int] = None

    # freeform
    model:         Optional[str] = None
    budget_tokens: Optional[int] = None
    thinking:      Optional[bool] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    input_key:     Optional[str] = None
    output_key:    Optional[str] = None
    feedback_mode: Optional[str] = None   # "auto" | "none"

    # decision
    outcomes:       Optional[list[dict]] = None   # [{value, next_step, description?}]
    is_loop_back:   Optional[bool] = None
    max_iterations: Optional[int] = None


class PipelineDefIn(BaseModel):
    """Wire-format for creating/updating a custom pipeline definition."""
    name:                 str
    description:          str = ""
    entry_step:            str
    steps:                 list[PipelineStepIn]
    edge_overrides:         dict[str, str] = {}
    max_total_iterations:   int = 20


class ClarifyRequest(BaseModel):
    answer: str


class RetryTruncatedRequest(BaseModel):
    """Body for POST /run/{run_uuid}/retry-truncated. output_cap is the
    new max_tokens ceiling to give the truncated node's next attempt —
    the frontend is expected to at least double whatever `cap` it read
    off the truncation payload (see GET /run's "truncation" field), but
    any positive value is accepted since the person may want to jump
    straight to a much larger number for a node that keeps re-truncating."""
    output_cap: int


class ChatMessageIn(BaseModel):
    message: str
    # Mirrors the explicit-toggle pattern used in RunRequest (see runs.js
    # "Skip ensemble" toggle) rather than having the server silently guess
    # intent from message content:
    #   replan=False (default) -> lighter path: reuse this run's already-
    #     checkpointed classification/plan_spec, re-enter around draft/bugfix.
    #   replan=True  -> full path: re-enter at classify with the whole chat
    #     history folded into normalised_input, as if task_type/profile could
    #     have changed based on the new message.
    replan: bool = False
    # Mirrors RunRequest.requested_profile/task_type (see runs.js's profile
    # segmented control, and runDetail.js's CHAT_PROFILES). Only meaningful
    # when replan=True — a non-replan turn never reaches classify_node, so
    # pinning these here would have nothing to apply to (see the 400 this
    # raises otherwise in chat.post_chat_message). None/"" = auto, same
    # convention as RunRequest.
    requested_profile: Optional[str] = None   # "short" | "medium" | "long" | "ultra" | None
    task_type:         Optional[str] = None   # "coding" | "ideation" | "mixed" | "describe" | None
    # Mirrors RunRequest.human_in_the_loop. Same replan=True gating as
    # requested_profile/task_type above — a non-replan turn never reaches
    # classify_node/plan_node, so there's nothing for this to apply to.
    human_in_the_loop: bool = True
    # Mirrors RunRequest.use_search — was previously entirely absent from
    # this schema, which meant a chat follow-up had no way to request
    # search grounding at all, regardless of what the first message asked
    # for: post_chat_message had no req.use_search to read, so turn_state
    # (built in _run_chat_replan) never set the key, and
    # state.get("use_search") in plan_node/ideation_node silently returned
    # None/falsy — no error, no warning, just a quietly-skipped search
    # branch. Unlike mode/task_type, this is NOT gated behind replan=True:
    # both plan_node and ideation_node read use_search directly off
    # whatever state _run_chat_replan builds, and that function is the
    # ONLY state-construction path for a chat turn today (the non-replan
    # "lighter path" is a documented TODO stub that currently just calls
    # _run_chat_replan too — see chat.py's _run_chat_turn_thread), so this
    # applies to every chat follow-up regardless of the replan flag.
    use_search: bool = False


class BudgetPatch(BaseModel):
    budgets: dict[str, int]   # stage → token budget (-1 = unlimited)


class ChessRequest(BaseModel):
    movePlayed:    Optional[str]   = None
    side:          Optional[str]   = None
    moveNotation:  Optional[str]   = None
    classification:Optional[str]  = None
    cpLoss:        Optional[int]   = None
    bestMove:      Optional[str]   = None
    bestMoveEval:  Optional[float] = None
    evalAfter:     Optional[float] = None
    gamePhase:     Optional[str]   = None
    winPctWhite:   Optional[float] = None
    winPctDraw:    Optional[float] = None
    winPctBlack:   Optional[float] = None
    materialDelta: Optional[int]   = None
    depthProfile:  Optional[str]   = None
    tacticalFlags: list[str]       = []
    bestLine:      list[str]       = []
    pieces:        Optional[dict]  = None
    slow_mode:     bool            = False
