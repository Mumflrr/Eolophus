"""
api/server.py — FastAPI server for the Eolophus pipeline.

Exposes all pipeline functionality over HTTP so the web UI, macOS SwiftUI app,
and the chess Bluetooth bridge can all talk to the same localhost server.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Dependencies (add to requirements.txt):
    fastapi uvicorn[standard] sse-starlette psutil
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

API_DIR      = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent
RUNS_DIR     = Path(os.environ.get("PIPELINE_RUNS_DIR", PROJECT_ROOT / "runs"))
STATIC_DIR   = API_DIR / "static"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── SearXNG (Docker) ─────────────────────────────────────────────────────────
# Best-effort only — mirrors clients/search.py's own tolerance of SearXNG
# being unreachable (search_web() never raises, just returns [] and logs a
# warning). server.py shouldn't be any stricter about it than the client
# that actually depends on it, so nothing here ever blocks or fails
# startup/shutdown. This does NOT install Docker or SearXNG — that's still
# setup_searxng.sh, run once by hand. This only brings up (and optionally
# tears down) an already-installed container, the same docker-compose
# commands start_all.sh's start_searxng() and stop_all.sh already run —
# just triggered by uvicorn instead of requiring those scripts.
_SEARXNG_COMPOSE_FILE = PROJECT_ROOT / "servers/docker-compose.searxng.yml"


def _searxng_compose(*args: str) -> bool:
    """Run `docker-compose -f <file> <args>`, logging but never raising.

    Uses the hyphenated v1 CLI (docker-compose), not the v2 `docker compose`
    plugin subcommand — matches what's actually installed (see docker.io +
    docker-compose 1.29.2 on this box) and what start_all.sh/stop_all.sh
    already invoke. Switch both sides together if you later move to v2.
    """
    if not _SEARXNG_COMPOSE_FILE.exists():
        log.info(
            "SearXNG: %s not found — skipping (%s). Run setup_searxng.sh once "
            "if you want web search available.",
            _SEARXNG_COMPOSE_FILE.name, " ".join(args),
        )
        return False
    try:
        result = subprocess.run(
            ["docker-compose", "-f", str(_SEARXNG_COMPOSE_FILE), *args],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.warning(
                "SearXNG: `docker-compose %s` failed (rc=%d): %s",
                " ".join(args), result.returncode, result.stderr.strip(),
            )
            return False
        log.info("SearXNG: docker-compose %s OK", " ".join(args))
        return True
    except FileNotFoundError:
        log.info("SearXNG: docker-compose not found on PATH — skipping (%s).", " ".join(args))
        return False
    except subprocess.TimeoutExpired:
        log.warning("SearXNG: `docker-compose %s` timed out after 30s.", " ".join(args))
        return False
    except Exception as e:
        log.warning("SearXNG: unexpected error running docker-compose %s: %s", " ".join(args), e)
        return False


# Whether to also stop the container on server shutdown. Off by default —
# `restart: unless-stopped` in the compose file is meant to let SearXNG
# outlive individual server restarts (see docker-compose.yml comments),
# so tearing it down here on every uvicorn --reload cycle would defeat
# that. Opt in explicitly if you want strict start/stop symmetry.
_SEARXNG_STOP_ON_SHUTDOWN = os.environ.get("SEARXNG_STOP_ON_SHUTDOWN", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: bring SearXNG up in the background so a slow/failed Docker
    # call can never delay uvicorn actually starting to serve requests.
    await asyncio.to_thread(_searxng_compose, "up", "-d")
    yield
    # Shutdown: opt-in only — see _SEARXNG_STOP_ON_SHUTDOWN above.
    if _SEARXNG_STOP_ON_SHUTDOWN:
        await asyncio.to_thread(_searxng_compose, "down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Eolophus Pipeline API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-run executor — one pipeline at a time on a single GPU
_executor   = ThreadPoolExecutor(max_workers=1)
_active_runs: dict[str, Any] = {}   # run_uuid → {future, env_overrides}
# _env_lock is the SAME lock object pipeline/graph.py's truncation-retry
# wrapper uses (imported from clients.llm, not redefined here) — see
# clients/llm.py's env_lock docstring for why this must be one shared
# instance rather than one threading.Lock() per module.
from clients.llm import env_lock as _env_lock


# ── Request / response schemas ─────────────────────────────────────────────────

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
    under (see _write_image).
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
                                            # created via the pipeline CRUD endpoints below.
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
    # raises otherwise in post_chat_message). None/"" = auto, same
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
    # _run_chat_replan too — see _run_chat_turn_thread), so this applies
    # to every chat follow-up regardless of the replan flag.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_run_dir(run_uuid: str) -> Path:
    return RUNS_DIR / run_uuid


def get_chat_turn_dir(run_uuid: str, seq: int) -> Path:
    """
    Per-turn artifact directory for a chat follow-up message.

    Each chat turn re-enters the graph, and every node writes its stage
    artifact (classification.json, draft.json, etc.) to Path(state["run_dir"])
    unconditionally — see nodes/*.py. If we handed the graph the same
    run_dir every turn, turn N's classify_node would silently overwrite
    turn N-1's classification.json, and so on for every stage file, so
    only the latest turn's artifacts ever survived.

    Giving each turn its own subdirectory (run_dir/turns/<seq>/) and
    passing THAT as run_dir into app_graph.invoke for chat turns fixes
    this for free — no node needs to change, they all just write into
    whatever run_dir they were handed. run.json (run-wide status) is
    deliberately NOT part of this — see _run_chat_turn_thread, which
    always writes run.json to the top-level run_dir, never the turn dir.
    """
    return get_run_dir(run_uuid) / "turns" / str(seq)


def _latest_turn_seq(run_uuid: str) -> Optional[int]:
    """Highest numbered subdirectory under run_dir/turns/, or None if there
    isn't one yet (i.e. this run has had no chat follow-ups)."""
    turns_dir = get_run_dir(run_uuid) / "turns"
    if not turns_dir.exists():
        return None
    seqs = [int(p.name) for p in turns_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(seqs) if seqs else None


def _active_sentinel_dir(run_uuid: str) -> Path:
    """
    Directory a currently-running graph invocation is actually writing
    its stage artifacts and halt sentinels (clarification.json,
    truncated.json) into — the top-level run_dir for the original run,
    or run_dir/turns/<latest_seq>/ once at least one chat follow-up has
    started (see get_chat_turn_dir / _run_chat_replan's turn_run_dir).

    This exists because clarify_node and the truncation-retry wrapper
    (pipeline/graph.py) both write their sentinel via Path(state["run_dir"]),
    and state["run_dir"] IS the turn directory for a chat follow-up, not
    the top-level run_dir — _run_status/_read_artifacts_from previously
    only ever checked the top-level directory, which happened to be
    invisible for clarification halts on turn 2+ (same latent bug this
    fixes for the new truncated.json sentinel) since nothing exercised
    that path loudly enough to notice: a clarification halt on a later
    turn would simply never report "waiting_for_clarification", it would
    fall through to whatever run.json's stale status said instead.
    """
    latest_seq = _latest_turn_seq(run_uuid)
    if latest_seq is not None:
        return get_chat_turn_dir(run_uuid, latest_seq)
    return get_run_dir(run_uuid)


def _status_after_invoke(run_uuid: str, final_state: dict, sentinel_dir: Optional[Path] = None) -> str:
    """
    Classify what app_graph.invoke()/Command(resume=...) just returned.

    A truthy final_state["__interrupt__"] means SOME node called
    interrupt() and the graph paused rather than finished — but that's
    true for both clarify_node's own interrupt() and the generic
    truncation-retry wrapper's interrupt() (pipeline/graph.py's
    _wrap_node_for_truncation_retry), and the two need different
    handling (different sentinel file, different chat message, different
    resume endpoint). Distinguish them by which sentinel is actually on
    disk in the directory this invocation just wrote into — same
    resolution _run_status uses for a later GET, kept consistent here so
    a status computed right after invoke() matches what a follow-up GET
    would report.

    sentinel_dir: pass the directory THIS invocation is actually writing
    into when the caller already knows it (e.g. /clarify's resume path,
    which pins it before starting the background resume — see that
    endpoint's comment for why re-deriving "latest" here would be wrong
    if a newer chat turn started while this invocation was running).
    Falls back to a fresh _active_sentinel_dir() lookup for callers that
    don't have a pinned directory (e.g. the original, non-resumed
    _run_pipeline_thread, where "latest turn" and "this run" are always
    the same directory since no follow-up turn exists yet).
    """
    if not final_state.get("__interrupt__"):
        return "unresolvable" if final_state.get("pipeline_failed") else "complete"
    active_dir = sentinel_dir if sentinel_dir is not None else _active_sentinel_dir(run_uuid)
    if (active_dir / "truncated.json").exists():
        return "waiting_for_truncation_retry"
    return "waiting_for_clarification"


def _post_halt_chat_message(run_uuid: str, run_dir: Path, status: str, sentinel_dir: Optional[Path] = None) -> None:
    """
    Surface a fresh halt (clarification or truncation) as an assistant
    chat message, same as _run_pipeline_thread has always done for
    clarification. For truncation, node_id="truncated" lets runDetail.js
    render a distinct "Retry with higher limit" affordance instead of the
    clarify answer box — see the truncation payload's stage/cap/
    thinking_block/partial_answer fields (pipeline/graph.py's
    _wrap_node_for_truncation_retry), which flow through here into the
    posted message's content and metadata_json so retryTruncated's UI
    has something concrete to show and act on.

    sentinel_dir: pass the directory the triggering invocation actually
    wrote its sentinel into, when the caller already knows it (see
    _status_after_invoke's docstring for why re-deriving "latest" via a
    fresh _active_sentinel_dir() call can pick the wrong directory if a
    newer chat turn started while a resume was still in flight).
    """
    from storage.chat_store import append_message
    active_dir = sentinel_dir if sentinel_dir is not None else _active_sentinel_dir(run_uuid)
    try:
        if status == "waiting_for_clarification":
            clar = _read_json(active_dir / "clarification.json") or {}
            question = clar.get("question", "Clarification needed.")
            append_message(run_uuid, role="assistant", content=question, node_id="clarify")
        elif status == "waiting_for_truncation_retry":
            trunc = _read_json(active_dir / "truncated.json") or {}
            stage      = trunc.get("stage", "unknown")
            cap        = trunc.get("cap")
            tokens_out = trunc.get("tokens_out")
            preview = (trunc.get("thinking_block") or trunc.get("partial_answer") or "").strip()
            if len(preview) > 4000:
                preview = preview[:4000] + "\n…[truncated for chat display — full text in truncated.json]"
            content = (
                f"Hit the token limit during '{stage}' (cap={cap}, "
                f"generated {tokens_out} tokens) before finishing.\n\n"
                + (f"What it was thinking:\n\n{preview}" if preview else "(No partial content was recovered.)")
            )
            append_message(run_uuid, role="assistant", content=content, node_id="truncated")
    except Exception as e:
        log.warning("Failed to record halt chat message for %s (%s): %s", run_uuid, status, e)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_status(run_uuid: str) -> str:
    """
    Derive run status from sentinel files in the run directory.

    Checks the LATEST turn directory first (the common case — an active
    invocation halts and nothing newer has started yet), but a halt
    sentinel written into an OLDER turn directory is still a genuine
    unresolved halt even after a newer turn exists: if a resume for turn N
    is still in flight when turn N+1 starts (see /clarify's
    resume_sentinel_dir comment for exactly this race), turn N's
    clarification.json/truncated.json won't be cleared until that resume
    actually completes, and a naive "check latest only" lookup would miss
    it entirely — silently dropping back to run.json's stale status
    instead of correctly reporting the still-pending halt. Scanning every
    turn directory is cheap (a handful of dirs, one small file check
    each) and turns_dir only grows across the life of one run's chat.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        return "not_found"

    active_dir = _active_sentinel_dir(run_uuid)
    if (active_dir / "clarification.json").exists():
        return "waiting_for_clarification"
    if (active_dir / "truncated.json").exists():
        return "waiting_for_truncation_retry"

    # Latest turn is clear — but an EARLIER turn's halt may still be
    # mid-resume (see docstring above). Check every turn dir, oldest to
    # newest is fine since we only care whether ANY is still halted.
    turns_dir = run_dir / "turns"
    if turns_dir.exists():
        for turn_path in turns_dir.iterdir():
            if not (turn_path.is_dir() and turn_path.name.isdigit()):
                continue
            if turn_path == active_dir:
                continue  # already checked above
            if (turn_path / "clarification.json").exists():
                return "waiting_for_clarification"
            if (turn_path / "truncated.json").exists():
                return "waiting_for_truncation_retry"

    rj = _read_json(run_dir / "run.json")
    if rj:
        return rj.get("status", "running")
    if run_uuid in _active_runs:
        fut = _active_runs[run_uuid].get("future")
        if fut and fut.running():
            return "running"
        if fut and fut.done():
            return "complete" if not fut.exception() else "error"
    return "unknown"


def _get_runs_from_db(limit: int = 100) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _get_lessons_from_db(
    task_type: Optional[str] = None,
    issue_category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 200,
) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        conditions = ["confidence_score >= ?"]
        params: list[Any] = [min_confidence]
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if issue_category:
            conditions.append("issue_category = ?")
            params.append(issue_category)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM lessons WHERE {where} ORDER BY confidence_score DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result
    finally:
        conn.close()


def _load_routing_config() -> dict:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _save_routing_config(cfg: dict) -> None:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── Pipeline runner (runs in ThreadPoolExecutor) ──────────────────────────────

def _run_pipeline_thread(
    run_uuid: str, initial_state: dict, env_overrides: dict,
    pipeline_name: Optional[str] = None,
) -> dict:
    """
    Synchronous pipeline runner. Executed in a background thread.
    env_overrides: dict of env vars to set for the duration of this run.
    pipeline_name: None runs the built-in pipeline (pipeline.graph.get_graph).
                   A string runs the named custom pipeline
                   (pipeline.custom_graph.get_custom_graph) — compiled fresh
                   from config/pipelines/{pipeline_name}.yaml, recompiled
                   automatically if that file's mtime changed since the
                   last run (edits made via the CRUD endpoints below take
                   effect on the NEXT run with no server restart needed).
    """
    run_dir = Path(initial_state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with _env_lock:
        # Save original values and set overrides
        saved = {k: os.environ.get(k) for k in env_overrides}
        for k, v in env_overrides.items():
            os.environ[k] = v

    try:
        from storage.critique_store import update_run_status

        if pipeline_name:
            from pipeline.custom_graph import get_custom_graph
            app_graph = get_custom_graph(pipeline_name)
            callbacks = []
        else:
            from pipeline.graph import get_graph
            app_graph, callbacks = get_graph()

        config = {"configurable": {"thread_id": run_uuid}}
        if callbacks:
            config["callbacks"] = callbacks

        _write_run_json(run_dir, run_uuid, initial_state.get("mode"), "running")
        final_state = app_graph.invoke(initial_state, config=config)

        # A node that called interrupt() (clarify_node, or the generic
        # truncation-retry wrapper in pipeline/graph.py) makes invoke()
        # return early with an "__interrupt__" entry in the returned
        # state instead of running to completion — the graph is paused,
        # not finished. _status_after_invoke checks BEFORE looking at
        # pipeline_failed and distinguishes clarify vs truncation halts
        # by which sentinel file is on disk (see its docstring) — a halt
        # of either kind must not be misreported as unresolvable/complete.
        status = _status_after_invoke(run_uuid, final_state)

        # DELETE /run/{uuid} on an active run sets status="cancelled" and
        # returns immediately — it can't hard-kill mid-inference (calls are
        # atomic per stage, see remove_run's docstring). If that happened
        # while app_graph.invoke() above was still running its current
        # stage, we'd otherwise unconditionally overwrite "cancelled" back
        # to "complete"/"unresolvable"/etc. right here — which is exactly
        # why a second DELETE (meant to hard-delete a cancelled run) could
        # keep re-cancelling an apparently-active run instead: this thread
        # kept clobbering the cancelled status after the fact. Check what's
        # actually on disk right now before writing a terminal status.
        current = _read_json(run_dir / "run.json") or {}
        if current.get("status") == "cancelled":
            log.info("Run %s was cancelled mid-flight — discarding pipeline result", run_uuid)
            return final_state

        _write_run_json(
            run_dir, run_uuid, initial_state.get("mode"), status,
            profile=final_state.get("profile"),
        )
        update_run_status(run_uuid, status)

        if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
            _post_halt_chat_message(run_uuid, run_dir, status)
        else:
            # Mirror _run_chat_turn_thread: persist the assistant's reply as
            # a chat turn once the run reaches a terminal state.
            try:
                from storage.chat_store import append_message
                reply_text = _extract_reply_text(final_state)
                append_message(
                    run_uuid, role="assistant", content=reply_text,
                    node_id=None, run_iteration=final_state.get("iteration"),
                )
            except Exception as e:
                log.warning("Failed to record initial-run assistant reply for %s: %s", run_uuid, e)

        return final_state

    except Exception as exc:
        log.exception("Pipeline run %s failed with an uncaught exception", run_uuid)
        _write_run_json(
            run_dir, run_uuid, initial_state.get("mode"), "error",
            error_detail=f"{type(exc).__name__}: {exc}",
        )
        try:
            from storage.critique_store import update_run_status
            update_run_status(run_uuid, "error")
        except Exception:
            pass
        raise

    finally:
        with _env_lock:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _write_run_json(
    run_dir: Path, run_uuid: str, mode: Optional[str], status: str,
    error_detail: Optional[str] = None, profile: Optional[str] = None,
) -> None:
    # Guards against a stuck background thread waking up (e.g. finishing a
    # very long LLM call) after DELETE /run/{run_uuid} has already
    # rmtree'd this run — see remove_run's cancelled-run branch. Without
    # this check, the write below would silently recreate run_dir with a
    # single orphaned run.json in it, making a deleted run reappear.
    if not run_dir.exists():
        log.info(
            "_write_run_json: run_dir for %s no longer exists (likely deleted "
            "after cancellation) — discarding status write '%s'", run_uuid, status,
        )
        return
    path = run_dir / "run.json"
    data = _read_json(path) or {}
    data.update({"run_uuid": run_uuid, "mode": mode, "status": status})
    # profile (resolved_profile from classify_node) is only known once the
    # graph has actually run classify — every call site before that point
    # passes profile=None, which must NOT stomp a previously-written value
    # on a later status-only write (e.g. a resume's "running" write before
    # the graph re-executes classify_node). runDetail.js reads this as
    # r.artifacts.run.profile to drive the ultra-ambient effect.
    if profile is not None:
        data["profile"] = profile
    # error_detail carries the actual exception message when status="error"
    # — previously every "except Exception as exc: _write_run_json(...,
    # 'error')" call site caught exc, wrote status='error' with NO detail
    # at all, then re-raised into a background ThreadPoolExecutor future
    # nothing ever reads the result of. GET /run (and therefore anyone
    # debugging from its output, e.g. by curling it) had no error message
    # to show, only the bare word "error" — a real exception happened but
    # was completely unrecoverable through any API surface. Only set (not
    # cleared) when provided, so a later non-error status write via the
    # same helper (e.g. a subsequent successful retry) doesn't need to
    # remember to explicitly clear a stale one — see the explicit
    # data.pop below for the one case that DOES need clearing.
    if error_detail is not None:
        data["error_detail"] = error_detail
    elif status != "error":
        data.pop("error_detail", None)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Run management endpoints ───────────────────────────────────────────────────
@app.get("/debug/run/{run_uuid}/exception")
async def debug_run_exception(run_uuid: str):
    info = _active_runs.get(run_uuid)
    if not info:
        return {"error": "not in _active_runs (may have been cleaned up)"}
    fut = info.get("future")
    if not fut or not fut.done():
        return {"error": "not done yet"}
    exc = fut.exception()
    if exc is None:
        return {"error": "no exception — finished cleanly"}
    import traceback
    return {"exception": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}

# _compose_input_with_attachments and its MAX_ATTACHMENT_CHARS cap now live in
# pipeline/attachments.py, so sub_spec_runner_node (pipeline/graph.py) can
# reuse the identical fencing/truncation logic when building each sub-spec's
# task_input, without pipeline/graph.py importing this module (server.py
# already imports pipeline.graph, so the reverse import would be circular).
# Aliased back to the original private names so every existing call site in
# this file is unchanged.
from pipeline.attachments import (
    MAX_ATTACHMENT_CHARS,
    compose_input_with_attachments as _compose_input_with_attachments,
)


def _write_attachments(run_dir: Path, attachments: list["AttachmentIn"]) -> list[dict]:
    """Persist attachments to run_dir/attachments/ and return manifest entries
    for run.json / the artifacts endpoint. Filenames are sanitized to a bare
    basename so a crafted filename can't escape run_dir.

    Each manifest entry gets excluded=False initially — see
    DELETE /run/{run_uuid}/attachments/{filename}, which flips this flag
    rather than deleting the file outright, so an excluded attachment can
    still be inspected in the advanced/artifacts view even though it's no
    longer folded into the pipeline's input on later turns."""
    if not attachments:
        return []
    att_dir = run_dir / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a in attachments:
        safe_name = Path(a.filename).name or "unnamed.txt"
        # de-dupe if two attachments share a basename
        dest = att_dir / safe_name
        i = 1
        while dest.exists():
            stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
            dest = att_dir / f"{stem}_{i}{suffix}"
            i += 1
        dest.write_text(a.content, encoding="utf-8")
        manifest.append({
            "filename":    safe_name,
            "size_bytes":  len(a.content.encode("utf-8")),
            "path":        str(dest.relative_to(run_dir)),
            "excluded":    False,
        })
    return manifest


def _load_live_attachments(run_dir: Path) -> list[dict]:
    """
    Read attachments.json + the actual file contents off disk, returning
    only the ones NOT excluded, as plain {filename, content} dicts ready
    for _compose_input_with_attachments.

    This is the fix for attachments silently dropping out of context after
    the first chat turn: previously, only start_run ever called
    _compose_input_with_attachments (from the request body, which only
    exists on that first call) — a replan's chat_input was built purely
    from chat history, so the model stopped seeing file content entirely
    after turn 1. Every turn now calls this instead, so attachments stay
    referenceable for the life of the run, until explicitly excluded.
    """
    manifest = _read_json(run_dir / "attachments.json")
    if not manifest:
        return []
    live = []
    for entry in manifest:
        if entry.get("excluded"):
            continue
        file_path = run_dir / entry["path"]
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            log.warning("Attachment file missing on disk, skipping: %s", file_path)
            continue
        live.append({"filename": entry["filename"], "content": content})
    return live


_ALLOWED_IMAGE_TYPES = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/jpg":  "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}


def _write_image(run_dir: Path, image: "ImageIn") -> str:
    """
    Decode a data URL and persist it as run_dir/image.<ext>, returning the
    absolute path. vision_decode_node (nodes/vision.py) re-derives the
    file extension from this same path via Path(image_path).suffix, so
    the extension written here must match the actual image bytes, not
    just default to whatever the browser called the file.

    Raises HTTPException(400) for anything that isn't a data: URL with a
    supported image mime type, or isn't valid base64 — fail fast in the
    request handler rather than let vision_decode_node discover a garbage
    file mid-run.
    """
    import base64
    import re

    m = re.match(r"^data:([\w/+.-]+);base64,(.+)$", image.data_url, re.DOTALL)
    if not m:
        raise HTTPException(
            status_code=400,
            detail="image.data_url must be a base64 data URL "
                   "(e.g. 'data:image/png;base64,...')",
        )
    mime_type, b64_data = m.group(1).lower(), m.group(2)
    ext = _ALLOWED_IMAGE_TYPES.get(mime_type)
    if not ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{mime_type}'. "
                   f"Supported: {', '.join(sorted(_ALLOWED_IMAGE_TYPES))}",
        )
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image.data_url is not valid base64")

    dest = run_dir / f"image.{ext}"
    dest.write_bytes(raw)
    return str(dest)


@app.get("/run/{run_uuid}/image")
async def get_run_image(run_uuid: str):
    """
    Serve the run's uploaded image bytes (see _write_image / ImageIn).
    Not served via the /static mount since RUNS_DIR isn't under it —
    this is a plain FileResponse instead, with the mime type re-derived
    from the extension _write_image chose.
    """
    from fastapi.responses import FileResponse

    run_dir = get_run_dir(run_uuid)
    matches = list(run_dir.glob("image.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="This run has no image")

    ext_to_mime = {v: k for k, v in _ALLOWED_IMAGE_TYPES.items() if k != "image/jpg"}
    path = matches[0]
    media_type = ext_to_mime.get(path.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(path, media_type=media_type)


@app.post("/run")
async def start_run(req: RunRequest):
    """Start a new pipeline run. Returns immediately with run_uuid."""
    run_uuid = str(uuid.uuid4())
    run_dir  = get_run_dir(run_uuid)
    run_dir.mkdir(parents=True, exist_ok=True)

    attachment_manifest = _write_attachments(run_dir, req.attachments)
    if attachment_manifest:
        (run_dir / "attachments.json").write_text(
            json.dumps(attachment_manifest, indent=2), encoding="utf-8"
        )

    # Image goes through vision_decode_node instead of the text-attachment
    # path — see route_after_input in pipeline/graph.py, which enters at
    # vision_decode whenever raw_image_path is set on state, then falls
    # through to classify once vision_decode_node has normalised the image
    # into text. Nothing downstream of that node ever sees the image itself.
    raw_image_path: Optional[str] = None
    if req.image:
        raw_image_path = _write_image(run_dir, req.image)

    composed_input = _compose_input_with_attachments(
        req.task, [a.dict() for a in req.attachments]
    )

    # vision_decode_node (nodes/vision.py) builds its OWN normalised_input
    # from state["raw_text_input"] and returns that as the final
    # normalised_input for the rest of the graph — it has no idea
    # composed_input (task + attachment text) exists. If raw_text_input
    # stayed as the bare task, attached file content would silently
    # disappear whenever an image is also present, since vision_decode's
    # returned normalised_input would overwrite the attachment-aware one
    # set below. Feed it the composed text instead so attachments survive
    # the vision_decode -> classify hop the same as they do without an image.
    text_for_pipeline = composed_input if raw_image_path else req.task

    # Persist the original task as the first chat turn. A chat is 1:1 with
    # a run (chat_uuid == run_uuid, see api.js), but previously only
    # follow-up turns (POST /chat) ever called append_message — the run's
    # own starting task/response never made it into chat_store, so
    # GET /chat/{run_uuid} came back empty until a user sent a follow-up.
    # Write the user turn now (task is known immediately); the assistant's
    # reply is appended once _run_pipeline_thread finishes below.
    try:
        from storage.chat_store import append_message
        append_message(run_uuid, role="user", content=req.task)
    except Exception as e:
        log.warning("Failed to record initial chat turn for %s: %s", run_uuid, e)

    initial_state: dict = {
        "run_uuid":          run_uuid,
        "run_dir":           str(run_dir),
        "iteration":         0,
        "is_sub_spec":       False,
        "decompose":         False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
        "raw_text_input":    text_for_pipeline,
        "raw_image_path":    raw_image_path,
        "normalised_input":  composed_input,
        "attachments":       [a.dict() for a in req.attachments],
        "use_search":        req.use_search,
        "human_in_the_loop": req.human_in_the_loop,
    }

    # env_overrides retained as reusable plumbing for any future flag that
    # genuinely needs a process-wide env var for the duration of a run
    # (hence the _env_lock around concurrent runs stomping on each other) —
    # but it no longer carries PIPELINE_ULTRA/PIPELINE_FORCE_SHORT/
    # PIPELINE_NO_ENSEMBLE. Those three are retired: profile selection now
    # flows through requested_profile in state, read directly by
    # classify_node/select_profile (see pipeline/routers.py), not through
    # env vars read by llm.py/routers.py.
    env_overrides: dict[str, str] = {}

    if req.pipeline:
        # Custom pipeline: requested_profile/task_type are concepts
        # belonging to the BUILT-IN pipeline's routers (route_after_classify,
        # route_after_bugfix, etc.) — a custom pipeline has none of those
        # routers, so applying these overrides would be silently meaningless
        # at best. Validate the pipeline exists now (fail fast, not after
        # the background thread starts) rather than discovering a typo'd
        # name only once the thread's exception handler fires.
        from pipeline.custom_graph import PIPELINES_DIR
        pipeline_path = PIPELINES_DIR / f"{req.pipeline}.yaml"
        if not pipeline_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No custom pipeline named '{req.pipeline}'. "
                       f"List available pipelines at GET /pipelines.",
            )
    else:
        if req.requested_profile and req.requested_profile != "auto":
            initial_state["requested_profile"] = req.requested_profile
        if req.task_type:
            initial_state["task_type"] = req.task_type

    # Write initial run.json immediately so the UI can poll it. profile
    # isn't known yet (classify_node hasn't run) — only mode was ever
    # meaningful this early, and mode itself isn't set until classify_node
    # returns it either; this call just seeds run_uuid/status.
    _write_run_json(run_dir, run_uuid, None, "running")

    # Register in DB
    try:
        from storage.critique_store import write_run
        write_run(
            run_uuid    = run_uuid,
            # Placeholder, same as before: real mode isn't known until
            # classify_node returns it. NOTE: storage/critique_store.py
            # wasn't available to confirm whether this DB column should
            # be renamed/extended for requested_profile — left as mode
            # only, unchanged in shape, to avoid guessing at its schema.
            mode        = req.pipeline or "auto",
            task_type   = req.task_type or "auto",
            complexity  = "auto",
            is_sub_spec = False,
        )
    except Exception as e:
        log.warning("Failed to write run to DB: %s", e)

    future = _executor.submit(
        _run_pipeline_thread, run_uuid, initial_state, env_overrides, req.pipeline
    )
    # env_overrides is stored here so a later POST /clarify on this run can
    # re-apply whatever env vars this run started with when it resumes —
    # see clarify_run below. Previously this dict never made it into
    # _active_runs, so clarify_run's lookup always saw {}.
    _active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }

    return {
        "run_uuid": run_uuid,
        "status":   "running",
        "run_dir":  str(run_dir),
        "pipeline": req.pipeline or "built-in",
    }



# ── Chat endpoints ───────────────────────────────────────────────────────────
# A chat is 1:1 with a run: chat_uuid == run_uuid (see api.js). POST /run
# creates both implicitly (the first "turn" is the original task). These
# endpoints handle FOLLOW-UP turns after that run reaches a terminal state.
#
# Lesson integration (per product requirement):
#   - On every chat turn, BEFORE re-entering the graph, retrieve relevant
#     lessons and record which ones were used against this specific turn
#     (storage/lesson_store.record_lesson_usage) — independent of the run
#     itself, so this survives the run/chat being deleted later, same as
#     lesson writes already do.
#   - Lesson WRITES on a chat turn reuse the existing distiller_node path
#     when the turn went through validate (both replan and lighter paths
#     eventually reach validate/distiller). The separate user-correction
#     detection path (chat_distiller_node) is not yet wired in — see the
#     stub and comment in nodes/distiller.py.

@app.get("/chat/{run_uuid}")
async def get_chat(run_uuid: str):
    """Full message history for a chat, oldest-first, plus which lessons
    were used on each assistant turn (for the runDetail.js sidebar)."""
    from storage.chat_store import get_messages
    from storage.lesson_store import get_lessons_used_for_chat

    run_dir = get_run_dir(run_uuid)
    messages = get_messages(run_uuid)
    if not messages and not run_dir.exists():
        raise HTTPException(status_code=404, detail="No chat or run found for this id")

    lessons_by_seq = get_lessons_used_for_chat(run_uuid)
    for m in messages:
        used = lessons_by_seq.get(m["seq"])
        if used:
            m["lessons_used"] = used

    return {"run_uuid": run_uuid, "messages": messages}


@app.post("/chat/{run_uuid}")
async def post_chat_message(run_uuid: str, req: ChatMessageIn):
    """
    Send a follow-up chat message. 409s if the underlying run is still
    in-flight or paused for clarification — those go through POST /clarify
    instead (see api.js comment on sendChatMessage).
    """
    from storage.chat_store import append_message, get_messages, format_history_for_prompt

    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    status = _run_status(run_uuid)
    if status in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="This run is still busy — wait for it to finish, or answer "
                   "the pending clarification via POST /clarify first.",
        )

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    if (req.requested_profile or not req.human_in_the_loop) and not req.replan:
        raise HTTPException(
            status_code=400,
            detail="requested_profile/human_in_the_loop can only be set when "
                   "replan=true — a non-replan turn never reaches "
                   "classify_node/plan_node, so there's nothing to pin them to.",
        )

    user_seq = append_message(run_uuid, role="user", content=message)

    run_json = _read_json(run_dir / "run.json") or {}
    pipeline_name = run_json.get("pipeline")  # None for built-in runs

    _write_run_json(run_dir, run_uuid, run_json.get("mode"), "running")
    try:
        from storage.critique_store import update_run_status
        update_run_status(run_uuid, "running")
    except Exception:
        pass

    history = get_messages(run_uuid)  # includes the turn we just appended

    # env_overrides preserves whatever the original run was started with
    # (see the env_overrides comment on _active_runs in start_run above) so
    # a clarification raised mid-chat still resumes with the same env vars.
    # Retired: the old per-turn PIPELINE_ULTRA/PIPELINE_FORCE_SHORT override
    # block — req.requested_profile now flows into turn_state directly (see
    # _run_chat_replan) and is read by classify_node/select_profile, not by
    # env vars read from llm.py/routers.py.
    env_overrides = dict(_active_runs.get(run_uuid, {}).get("env_overrides", {}))

    future = _executor.submit(
        _run_chat_turn_thread, run_uuid, str(run_dir), history, req.replan, pipeline_name, user_seq,
        req.requested_profile, req.task_type, req.human_in_the_loop, env_overrides, req.use_search,
    )
    _active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }

    return {"status": "running", "run_uuid": run_uuid, "user_message_seq": user_seq}


@app.delete("/chat/{run_uuid}")
async def remove_chat(run_uuid: str):
    """
    Delete a chat's messages AND the per-turn artifact directories that
    back them (run_dir/turns/*, see get_chat_turn_dir). Mirrors
    chat_store.delete_chat's own contract: lessons and lesson_usage rows
    are untouched, since lessons are deliberately chat-agnostic (see
    schema_additions.sql) and must survive this.

    Does NOT touch the top-level run artifacts (run.json, classification.json,
    etc. sitting directly in run_dir) or the run row itself — those belong
    to the original run, not the chat, and are removed only via
    DELETE /run/{run_uuid}.

    409s under the same condition as POST /chat: if a chat turn is
    currently in flight, deleting out from under it could delete a turn
    directory a background thread is mid-write to.
    """
    from storage.chat_store import delete_chat

    status = _run_status(run_uuid)
    if status in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="This run is still busy — wait for it to finish before deleting the chat.",
        )

    deleted_count = delete_chat(run_uuid)

    turns_dir = get_run_dir(run_uuid) / "turns"
    removed_dirs = 0
    if turns_dir.exists():
        import shutil
        for seq_dir in turns_dir.iterdir():
            if seq_dir.is_dir():
                try:
                    shutil.rmtree(seq_dir)
                    removed_dirs += 1
                except FileNotFoundError:
                    pass
        # Clean up the now-empty turns/ dir itself too.
        try:
            turns_dir.rmdir()
        except OSError:
            pass  # not empty (race with a concurrent write) or already gone

    log.info(
        "Chat deleted: run=%s (%d messages, %d turn artifact dirs)",
        run_uuid, deleted_count, removed_dirs,
    )
    return {
        "status":            "deleted",
        "run_uuid":          run_uuid,
        "messages_deleted":  deleted_count,
        "turn_dirs_removed": removed_dirs,
    }


def _run_chat_turn_thread(
    run_uuid:            str,
    run_dir:             str,
    history:             list[dict],
    replan:              bool,
    pipeline_name:       Optional[str],
    turn_seq:            int,
    requested_profile:   Optional[str] = None,
    task_type_override:  Optional[str] = None,
    human_in_the_loop:   bool = True,
    env_overrides:       Optional[dict] = None,
    use_search:          bool = False,
) -> dict:
    """
    Background thread for a chat follow-up turn. Retrieves relevant lessons
    up front (regardless of replan/lighter path — see module-level comment),
    re-enters the graph, then persists the assistant reply as a chat turn
    and records which lessons were actually used against it.

    turn_seq is the user message's chat seq (assigned in post_chat_message)
    and is used to give this turn's stage artifacts their own directory —
    see get_chat_turn_dir. run_dir here stays the top-level run directory:
    it's still what run.json status writes target, and it's what the
    graph's OWN run_dir gets derived from for this turn.

    requested_profile/task_type_override mirror RunRequest.requested_profile/
    task_type, pinned into turn_state the same way start_run pins them into
    initial_state (only meaningful when replan=True — see the 400 raised in
    post_chat_message otherwise). human_in_the_loop mirrors
    RunRequest.human_in_the_loop the same way. Retired: the old per-turn
    PIPELINE_ULTRA/PIPELINE_FORCE_SHORT env-var window — profile now flows
    through turn_state/requested_profile, read directly by classify_node.
    env_overrides here is just whatever the run's persisted env vars are
    (see post_chat_message), applied for this thread's duration same as
    _run_pipeline_thread, with nothing chat-turn-specific added to it.

    use_search mirrors RunRequest.use_search (see ChatMessageIn.use_search's
    docstring for why this was missing entirely before and what that broke:
    plan_node/ideation_node would silently skip their search_web() call on
    every chat follow-up, without ever logging anything, since
    state.get("use_search") just came back None). Passed straight through
    to _run_chat_replan's turn_state regardless of replan, since that's the
    only state-construction path a chat turn goes through today.
    """
    from storage.chat_store import append_message, format_history_for_prompt
    from storage.critique_store import update_run_status

    run_dir_path = Path(run_dir)
    turn_dir     = get_chat_turn_dir(run_uuid, turn_seq)
    turn_dir.mkdir(parents=True, exist_ok=True)
    history_text = format_history_for_prompt(history)

    # ── Lesson retrieval, up front, every turn ──────────────────────────
    relevant_lessons = []
    try:
        from schemas.lesson import LessonQuery
        from storage.lesson_store import retrieve_lessons
        run_json = _read_json(run_dir_path / "run.json") or {}
        task_type = run_json.get("task_type", "coding")
        relevant_lessons = retrieve_lessons(LessonQuery(
            task_type=task_type,
            tags=[],
            top_k=5,
            min_score=0.0,
        ))
    except Exception as e:
        log.warning("Chat turn: lesson retrieval failed for %s: %s", run_uuid, e)

    # Apply this run's persisted env vars for the duration of this thread
    # only — same save/restore pattern as _run_pipeline_thread above, kept
    # as its own block since a chat turn's window must not leak past this
    # thread's lifetime.
    env_overrides = env_overrides or {}
    saved_env = {}
    with _env_lock:
        saved_env = {k: os.environ.get(k) for k in env_overrides}
        for k, v in env_overrides.items():
            os.environ[k] = v

    try:
        if replan:
            final_state = _run_chat_replan(
                run_uuid, run_dir_path, str(turn_dir), history_text,
                pipeline_name, relevant_lessons,
                requested_profile=requested_profile, task_type_override=task_type_override,
                human_in_the_loop=human_in_the_loop,
                use_search=use_search,
            )
        else:
            # TODO(lighter path): re-enter the LangGraph checkpoint at
            # draft/draft_short instead of classify, reusing the
            # classification/plan_spec already sitting in MemorySaver
            # under thread_id=run_uuid from the original run. This is NOT
            # a plain app_graph.invoke() — LangGraph's conditional entry
            # point is fixed at compile time (see route_after_input in
            # graph.py), so a genuine "start partway through" re-entry
            # needs either a second entry point wired into get_graph(),
            # or driving it through the same interrupt()/Command(resume=)
            # mechanism clarify_node uses. Deferring rather than guessing
            # at LangGraph internals against nodes I haven't verified —
            # falls back to the full replan path for now so lighter-path
            # requests still work, just not more cheaply yet.
            log.warning(
                "Chat turn %s: lighter (non-replan) path not yet implemented, "
                "falling back to full replan.", run_uuid,
            )
            final_state = _run_chat_replan(
                run_uuid, run_dir_path, str(turn_dir), history_text, pipeline_name,
                relevant_lessons, use_search=use_search,
            )

        status = _status_after_invoke(run_uuid, final_state)

        # Same cancellation-race guard as _run_pipeline_thread: don't let a
        # chat turn that was still in-flight when the run got cancelled
        # clobber "cancelled" back to a terminal status after the fact.
        current = _read_json(run_dir_path / "run.json") or {}
        if current.get("status") == "cancelled":
            log.info("Chat turn on run %s was cancelled mid-flight — discarding result", run_uuid)
            return final_state

        _write_run_json(
            run_dir_path, run_uuid, final_state.get("mode"), status,
            profile=final_state.get("profile"),
        )
        update_run_status(run_uuid, status)

        if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
            # Previously this branch didn't exist at all: a halt mid-chat-
            # turn fell straight into the "complete" reply-extraction path
            # below, misreporting the run as finished and posting whatever
            # _extract_reply_text's fallback text happened to be instead
            # of the actual question/truncation details. _post_halt_chat_message
            # reads the sentinel from the TURN directory (via
            # _active_sentinel_dir), which is what run_dir=turn_dir here
            # means it will actually find.
            _post_halt_chat_message(run_uuid, run_dir_path, status)
            return final_state

        reply_text = _extract_reply_text(final_state)
        # No node currently stamps "which node produced this state" onto
        # PipelineState (checked graph.py/state.py — no such field exists).
        # Leaving node_id unset rather than inventing a key nothing writes to;
        # runDetail.js's hasDetail already falls back to lessons-only detail
        # when node_id is absent, so this doesn't break the UI, it just means
        # chat turns won't show a node pill until a node actually reports this.
        node_id = None
        assistant_seq = append_message(
            run_uuid, role="assistant", content=reply_text,
            node_id=node_id, run_iteration=final_state.get("iteration"),
        )

        if relevant_lessons:
            try:
                from storage.lesson_store import record_lesson_usage
                record_lesson_usage(run_uuid, assistant_seq, relevant_lessons)
            except Exception as e:
                log.warning("Failed to record lesson usage for %s seq %d: %s", run_uuid, assistant_seq, e)

        return final_state

    except Exception as exc:
        log.exception("Chat turn on run %s failed with an uncaught exception", run_uuid)
        _write_run_json(
            run_dir_path, run_uuid, None, "error",
            error_detail=f"{type(exc).__name__}: {exc}",
        )
        try:
            update_run_status(run_uuid, "error")
        except Exception:
            pass
        append_message(run_uuid, role="system", content=f"Chat turn failed: {exc}")
        raise
    finally:
        _active_runs.pop(run_uuid, None)
        with _env_lock:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _run_chat_replan(
    run_uuid:            str,
    run_dir_path:        Path,
    turn_run_dir:        str,
    history_text:        str,
    pipeline_name:       Optional[str],
    relevant_lessons:    list,
    requested_profile:   Optional[str] = None,
    task_type_override:  Optional[str] = None,
    human_in_the_loop:   bool = True,
    use_search:          bool = False,
) -> dict:
    """Full graph re-entry at classify, with full chat history folded into
    normalised_input. Reuses the same thread_id checkpoint as the original
    run — LangGraph will still see prior state, but classify_node etc. run
    fresh, which is the whole point of 'replan'.

    requested_profile/task_type_override, when set, are pinned into
    turn_state exactly like start_run pins RunRequest.requested_profile/
    task_type into initial_state. task_type_override maps to
    classify_node's pinned_task_type branch, same as before. requested_profile
    is a DIFFERENT concept from mode pinning (see classifier.py: mode
    pinning controls what the classifier itself is told to keep fixed;
    requested_profile picks the pipeline shape downstream of classification,
    via select_profile()/resolved_profile — see classify_node's "Profile
    resolution" comment) — it's passed straight through to turn_state and
    read there, not translated into a pinned "mode" here. Without this, a
    replanned chat turn could never be forced into e.g. "long", since
    classify_node's own select_profile() would always decide, the same gap
    that made runs.js's profile picker have no equivalent in chat follow-ups.

    human_in_the_loop mirrors RunRequest.human_in_the_loop into turn_state,
    same pin-through as requested_profile/task_type_override above — gates
    whether classify_node/plan_node halt to ask before escalating models
    for this turn, or auto-escalate/proceed best-effort with no one to ask.

    use_search mirrors RunRequest.use_search into turn_state, same as
    requested_profile/task_type_override above. Previously ABSENT from
    turn_state entirely — not defaulted to False, just never set as a key
    at all — so plan_node/ideation_node's state.get("use_search") read
    back None on every chat follow-up regardless of what the ORIGINAL
    run's RunRequest.use_search had been, or of a person re-enabling
    search on a later turn: there was no plumbing from ChatMessageIn
    through to here for it to even be a per-turn choice. This is why a
    person could ask a follow-up question and get "I can't search" with
    no error anywhere — search_web() was never being called, not failing.

    run_dir_path is the top-level run directory (where attachments.json and
    the attachments/ folder live) — kept separate from turn_run_dir (this
    turn's own stage-artifact directory, see get_chat_turn_dir) since
    attachments are run-scoped, not turn-scoped.

    Live (non-excluded) attachments are re-folded into chat_input on every
    call here — previously this only happened once, in start_run, from the
    original request body. A replan's input was built from chat history
    alone, so any attached file's content silently stopped being visible
    to the model after the very first turn. Excluding an attachment (see
    DELETE /run/{run_uuid}/attachments/{filename}) is what should make it
    stop appearing here, not the accident of it being turn 2+."""
    if pipeline_name:
        from pipeline.custom_graph import get_custom_graph
        app_graph = get_custom_graph(pipeline_name)
        callbacks = []
    else:
        from pipeline.graph import get_graph
        app_graph, callbacks = get_graph()

    config = {"configurable": {"thread_id": run_uuid}}
    if callbacks:
        config["callbacks"] = callbacks

    live_attachments = _load_live_attachments(run_dir_path)
    chat_input = (
        f"{history_text}\n\n"
        f"(Continue the task above based on the most recent user message.)"
    )
    chat_input = _compose_input_with_attachments(chat_input, live_attachments)

    turn_state: dict = {
        "run_dir":          turn_run_dir,
        "run_uuid":         run_uuid,
        "iteration":        0,
        "raw_text_input":   chat_input,
        "normalised_input": chat_input,
        "attachments":      live_attachments,
        "pipeline_complete": False,
        "pipeline_failed":   False,
        "relevant_lessons":  relevant_lessons,
        "use_search":        use_search,
        "human_in_the_loop": human_in_the_loop,
    }
    if requested_profile and requested_profile != "auto":
        turn_state["requested_profile"] = requested_profile
    if task_type_override:
        turn_state["task_type"] = task_type_override

    return app_graph.invoke(turn_state, config=config)


def _extract_reply_text(final_state: dict) -> str:
    """
    Best-effort extraction of "the text to show as the assistant's chat
    reply" from whatever artifact the graph actually produced this turn.
    Mirrors runDetail.js's NODE_ARTIFACT_KEY fallback logic: prefer a
    known final-output field, fall back to whatever's most informative.
    """
    # final_output_path is a *path* to final.json, not text — describe_node
    # (and presumably other terminal nodes) writes {"answer": ..., ...}
    # there. The old code returned the path string itself here, which is
    # why chat turns whose only output was final_output_path showed nothing
    # useful (or, combined with a truncated/empty answer, an empty string
    # once dereferenced downstream). Read the file and pull "answer" out.
    final_output_path = final_state.get("final_output_path")
    if final_output_path:
        data = _read_json(Path(final_output_path))
        if data and data.get("answer"):
            return data["answer"]
        # File exists but has no usable answer (e.g. describe_node's
        # truncation-with-no-content case) — fall through to failure_reason
        # below instead of returning the path or an empty string.

    for key in ("fixed_output", "draft_output"):
        val = final_state.get(key)
        if val:
            return val if isinstance(val, str) else getattr(val, "summary", None) or str(val)
    if final_state.get("failure_reason"):
        return f"Could not complete: {final_state['failure_reason']}"
    return "(no textual output produced this turn)"


@app.get("/runs")
async def list_runs(limit: int = 50):
    """List recent pipeline runs from the database."""
    db_runs = _get_runs_from_db(limit)
    # Enrich with live status for active runs
    for r in db_runs:
        uid = r.get("run_uuid")
        if uid in _active_runs:
            r["status"] = _run_status(uid)
    return db_runs


_ARTIFACT_FILENAMES = [
    "classification.json", "planspec.json", "draft.json",
    "appraisal_report.json", "audit.json", "fixed.json", "critique.json",
    "verdict.json", "final.json", "final_validation.json",
]


def _read_artifacts_from(dir_path: Path, include_run_and_attachments: bool = False) -> dict:
    """Shared reader for both the top-level run artifacts and each chat
    turn's artifacts — same filenames, same shape, just a different dir."""
    fnames = list(_ARTIFACT_FILENAMES)
    if include_run_and_attachments:
        fnames = ["run.json"] + fnames + ["attachments.json"]
    artifacts: dict = {}
    for fname in fnames:
        data = _read_json(dir_path / fname)
        if data is not None:
            artifacts[fname.replace(".json", "")] = data
    return artifacts


def _read_iteration_snapshots(dir_path: Path) -> dict:
    """
    Read run_dir/iterations/<n>/*.json (or turn_dir/iterations/<n>/*.json
    for a chat follow-up — see write_iteration_artifact's docstring for
    why this needs to be checked in whatever directory a node was
    actually given, same as the turn-directory resolution _read_artifacts_from
    already needs elsewhere). Returns {"0": {"fixed": {...}, "verdict": {...}},
    "1": {...}, ...} — only iterations that actually produced at least one
    of the loop-writable artifacts (fixed/verdict/draft/critique) appear;
    a single-pass run with no correction loop has no iterations/ directory
    at all and this returns {}.
    """
    iterations: dict = {}
    iterations_dir = dir_path / "iterations"
    if not iterations_dir.exists():
        return iterations
    for iter_dir in sorted(
        (p for p in iterations_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        snap = _read_artifacts_from(iter_dir)
        if snap:
            iterations[iter_dir.name] = snap
    return iterations


@app.get("/run/{run_uuid}")
async def get_run(run_uuid: str):
    """Full run detail: artifacts, stages log, current status, each chat
    turn's own artifact snapshot (see get_chat_turn_dir), and each
    correction-loop iteration's own snapshot of fixed/verdict/draft/
    critique (see write_iteration_artifact)."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts = _read_artifacts_from(run_dir, include_run_and_attachments=True)

    # {iteration: {fixed: {...}, verdict: {...}, ...}} for the TOP-LEVEL
    # run (i.e. the original, non-chat run's own correction loop, or a
    # chat follow-up's loop if it wrote straight into run_dir — which it
    # doesn't; see turn_iterations below for that case). artifacts still
    # shows only the LATEST iteration's fixed/verdict/etc, unchanged, so
    # existing consumers of artifacts.fixed/.verdict keep working exactly
    # as before — this is purely additive.
    iterations = _read_iteration_snapshots(run_dir)

    # Per-turn artifacts: {seq: {classification: {...}, draft: {...}, ...}}.
    # Every chat turn writes into its own run_dir/turns/<seq>/ (see
    # get_chat_turn_dir) instead of overwriting the top-level files above,
    # so this is what lets the UI show e.g. "classify.json for message 3"
    # distinctly from message 1's classify.json. Each turn can ALSO have
    # its own correction loop, so turn_iterations mirrors the top-level
    # iterations map but scoped per turn seq: {seq: {iteration: {...}}}.
    chat_artifacts: dict = {}
    turn_iterations: dict = {}
    turns_dir = run_dir / "turns"
    if turns_dir.exists():
        for seq_dir in sorted(turns_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if not seq_dir.is_dir():
                continue
            turn_artifacts = _read_artifacts_from(seq_dir)
            if turn_artifacts:
                chat_artifacts[seq_dir.name] = turn_artifacts
            turn_iters = _read_iteration_snapshots(seq_dir)
            if turn_iters:
                turn_iterations[seq_dir.name] = turn_iters

    stages: list[dict] = []
    stages_path = run_dir / "stages.log"
    if stages_path.exists():
        for line in stages_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    stages.append(json.loads(line))
                except Exception:
                    pass

    # clarification.json / truncated.json are written by whichever
    # directory the graph is CURRENTLY executing under — the top-level
    # run_dir for the original run, or run_dir/turns/<latest_seq>/ once a
    # chat follow-up is in flight (see _active_sentinel_dir's docstring
    # for why this can't just be run_dir unconditionally: clarify_node
    # and the truncation-retry wrapper both write via
    # Path(state["run_dir"]), which IS the turn dir for a chat follow-up).
    active_dir    = _active_sentinel_dir(run_uuid)
    clarification = _read_json(active_dir / "clarification.json")
    truncation    = _read_json(active_dir / "truncated.json")

    # Surface the uploaded image (if any) as a relative path the frontend
    # can build a static URL from — same pattern as attachments.json's
    # "path" field. Doesn't try every extension in _ALLOWED_IMAGE_TYPES
    # exhaustively via glob since _write_image only ever writes exactly
    # one image.* file per run.
    image_path = None
    matches = list(run_dir.glob("image.*"))
    if matches:
        image_path = str(matches[0].relative_to(run_dir))

    return {
        "run_uuid":        run_uuid,
        "status":          _run_status(run_uuid),
        "artifacts":       artifacts,
        "iterations":      iterations,
        "chat_artifacts":  chat_artifacts,
        "turn_iterations": turn_iterations,
        "image_path":      image_path,
        "stages":          stages,
        "clarification":   clarification,
        "truncation":      truncation,
    }


def _update_attachment_manifest(run_dir: Path, filename: str, excluded: bool) -> dict:
    """Shared implementation for exclude/re-include. Flips the excluded flag
    on the matching manifest entry and rewrites attachments.json. The file
    on disk is never touched — excluding an attachment only stops it being
    folded into future turns' input (see _load_live_attachments); it stays
    fully visible in the run's artifacts/advanced view either way, and can
    be re-included later without re-uploading it."""
    manifest_path = run_dir / "attachments.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        raise HTTPException(status_code=404, detail="This run has no attachments")

    match = next((e for e in manifest if e.get("filename") == filename), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"No attachment named '{filename}' on this run",
        )

    match["excluded"] = excluded
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return match


@app.post("/run/{run_uuid}/attachments")
async def add_attachments(run_uuid: str, req: "AddAttachmentsIn"):
    """
    Add one or more new text/code attachments to an already-started run,
    for use from an ongoing chat (runDetail.js) rather than only at
    creation time (runs.js's POST /run).

    Reuses _write_attachments for on-disk persistence (same de-dupe-by-
    basename behavior as the initial run), then merges the new entries
    into the existing attachments.json instead of overwriting it, so
    attachments added at run-start and mid-chat live in one manifest.
    Newly-added files are picked up on the *next* turn via
    _load_live_attachments — same mechanism as everything else in that
    manifest, so no separate wiring is needed for them to reach the model.

    Allowed even while the run is mid-turn: this only writes to disk and
    appends to the manifest, it doesn't touch _active_runs or submit
    anything to the executor, so there's no race with an in-flight turn
    the way there is for POST /chat.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    if not req.attachments:
        raise HTTPException(status_code=400, detail="No attachments provided")

    new_entries = _write_attachments(run_dir, req.attachments)

    manifest_path = run_dir / "attachments.json"
    manifest = _read_json(manifest_path) or []
    manifest.extend(new_entries)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"status": "added", "attachments": new_entries}


@app.delete("/run/{run_uuid}/attachments/{filename}")
async def exclude_attachment(run_uuid: str, filename: str):
    """
    Exclude an attachment from future turns without deleting it. Once
    excluded, _load_live_attachments skips it, so it stops being folded
    into normalised_input (and stops costing tokens) on the run's current
    turn and every turn after — but the file and its manifest entry are
    kept, so it still shows up in the run's artifacts view and can be
    brought back with POST .../attachments/{filename}/include.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    entry = _update_attachment_manifest(run_dir, filename, excluded=True)
    return {"status": "excluded", "attachment": entry}


@app.post("/run/{run_uuid}/attachments/{filename}/include")
async def include_attachment(run_uuid: str, filename: str):
    """Re-include a previously excluded attachment — the inverse of
    DELETE .../attachments/{filename}. No re-upload needed; the file was
    never removed from disk, only skipped when folding input."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    entry = _update_attachment_manifest(run_dir, filename, excluded=False)
    return {"status": "included", "attachment": entry}


@app.post("/clarify/{run_uuid}")
async def clarify_run(run_uuid: str, req: ClarifyRequest):
    """
    Resume a pipeline that has halted for clarification.

    Mirrors _run_pipeline_thread's bookkeeping, which this endpoint
    previously skipped entirely:
      - run_dir.mkdir(): defensive, same as _run_pipeline_thread — avoids
        a FileNotFoundError from llm.py's log writers if the directory
        was ever missing at resume time.
      - env_overrides applied under _env_lock: whatever env vars this run
        was started with were previously lost on resume, since
        _active_runs never stored them and nothing set os.environ here.
      - run.json/DB written to "running" before resuming, and to a
        terminal status after — previously neither happened, so the chat/
        run status stayed stuck on "waiting_for_clarification" in the UI
        even after the pipeline had actually resumed and finished.
      - exceptions from the background thread are now caught and recorded
        instead of vanishing silently into an unobserved Future.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    run_dir.mkdir(parents=True, exist_ok=True)   # in case it's mid-creation

    # Pin the sentinel directory NOW, before kicking off the background
    # resume thread below. _active_sentinel_dir() always resolves to
    # whatever the LATEST turn directory is *at the moment it's called* —
    # it has no notion of "which resume this call belongs to". The actual
    # app_graph.invoke(Command(resume=...)) call below can run for minutes
    # (bugfix/audit/validate are slow stages), and if a NEW chat turn
    # starts anywhere in that window — another message, another halt, any
    # code path that creates run_dir/turns/<n+1>/ — then a later call to
    # _active_sentinel_dir() (e.g. from a GET /run poll, or from
    # _status_after_invoke() below) will resolve to that newer turn's
    # directory instead of the one THIS resume is actually writing into.
    # That mismatch is what caused a run to look stuck on
    # "waiting_for_clarification" even after the person answered and the
    # graph had genuinely finished: the finished run's real output landed
    # in the turn directory captured here, while status-checking code
    # was — by the time anyone polled it — looking at a different,
    # unrelated turn directory that _active_sentinel_dir() now considered
    # "latest". Resolve once, pass the pinned path everywhere this
    # specific resume needs to check or clear a sentinel, and don't
    # re-derive "latest" partway through handling one resume.
    resume_sentinel_dir = _active_sentinel_dir(run_uuid)

    # Remove clarification sentinel from that pinned directory — NOT a
    # fresh _active_sentinel_dir() call, for the reason above.
    clarification_file = resume_sentinel_dir / "clarification.json"
    if clarification_file.exists():
        clarification_file.unlink()

    # Persist the answer as a chat turn right away. Previously nothing on
    # this endpoint ever called append_message, so neither the clarifying
    # question nor the answer nor the eventual reply showed up in
    # GET /chat/{run_uuid} — the run resumed correctly under the hood, but
    # the chat drawer had no way to know, so "submitting an answer" looked
    # like it did nothing.
    try:
        from storage.chat_store import append_message
        append_message(run_uuid, role="user", content=req.answer)
    except Exception as e:
        log.warning("Failed to record clarification answer for %s: %s", run_uuid, e)

    env_overrides = _active_runs.get(run_uuid, {}).get("env_overrides", {})
    run_json = _read_json(run_dir / "run.json") or {}

    # Synthetic stages.log entry marking the resume boundary. stages.log
    # is a flat, ever-appending log across every clarify round on this
    # run — with 3 rounds in a row it showed as three bare "Classify"
    # entries with nothing distinguishing them, so from the timeline
    # alone there was no way to tell these were 3 separate resumed
    # attempts rather than, say, retries of one call. runDetail.js's
    # stageNode() already has a precedent for a synthetic, specially-
    # rendered entry type (model_swap) — this follows the same pattern.
    try:
        from clients.llm import _log_stage_entry
        _log_stage_entry(
            run_dir=str(run_dir), stage="clarification_resumed", model_name="",
            prompt_hash="", tokens_in=0, tokens_out=0, latency_ms=0.0,
            status="ok",
        )
    except Exception as e:
        log.warning("Failed to write clarification_resumed marker for %s: %s", run_uuid, e)

    _write_run_json(run_dir, run_uuid, run_json.get("mode"), "running")
    try:
        from storage.critique_store import update_run_status
        update_run_status(run_uuid, "running")
    except Exception:
        pass

    def _resume():
        with _env_lock:
            saved = {k: os.environ.get(k) for k in env_overrides}
            for k, v in env_overrides.items():
                os.environ[k] = v

        try:
            from pipeline.graph import get_graph
            from langgraph.types import Command
            app_graph, callbacks = get_graph()
            config = {"configurable": {"thread_id": run_uuid}}
            if callbacks:
                config["callbacks"] = callbacks

            final_state = app_graph.invoke(Command(resume=req.answer), config=config)

            # Same "halt before pipeline_failed" ordering as
            # _run_pipeline_thread — a second halt (clarify or
            # truncation) on this resume must not be misreported as
            # unresolvable/complete. Pass the pinned resume_sentinel_dir
            # captured before this background thread started — see its
            # definition above for why re-deriving "latest" here could
            # silently point at an unrelated, newer chat turn's directory.
            status = _status_after_invoke(run_uuid, final_state, sentinel_dir=resume_sentinel_dir)

            # Same cancellation-race guard as _run_pipeline_thread: the run
            # could have been cancelled while this resume was mid-flight.
            current = _read_json(run_dir / "run.json") or {}
            if current.get("status") == "cancelled":
                log.info("Run %s was cancelled mid-resume — discarding result", run_uuid)
                return final_state

            _write_run_json(
                run_dir, run_uuid, final_state.get("mode"), status,
                profile=final_state.get("profile"),
            )
            try:
                from storage.critique_store import update_run_status
                update_run_status(run_uuid, status)
            except Exception:
                pass

            # Same append_message pattern as _run_pipeline_thread: a fresh
            # halt gets its question/truncation details posted as an
            # assistant turn; anything terminal gets the actual reply.
            if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
                _post_halt_chat_message(run_uuid, run_dir, status, sentinel_dir=resume_sentinel_dir)
            else:
                try:
                    from storage.chat_store import append_message
                    reply_text = _extract_reply_text(final_state)
                    append_message(
                        run_uuid, role="assistant", content=reply_text,
                        node_id=None, run_iteration=final_state.get("iteration"),
                    )
                except Exception as e:
                    log.warning("Failed to record post-resume chat turn for %s: %s", run_uuid, e)

            return final_state

        except Exception as exc:
            log.exception("Resume (clarify) on run %s failed with an uncaught exception", run_uuid)
            _write_run_json(
                run_dir, run_uuid, run_json.get("mode"), "error",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
            try:
                from storage.critique_store import update_run_status
                update_run_status(run_uuid, "error")
            except Exception:
                pass
            try:
                from storage.chat_store import append_message
                append_message(run_uuid, role="system", content=f"Resume failed: {exc}")
            except Exception:
                pass
            raise

        finally:
            with _env_lock:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    future = _executor.submit(_resume)
    _active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }
    return {"status": "resumed"}


@app.post("/run/{run_uuid}/retry-truncated")
async def retry_truncated(run_uuid: str, req: RetryTruncatedRequest):
    """
    Resume a pipeline that halted because a node's model call hit its
    output token cap (see clients.llm.TruncatedOutputError and
    pipeline/graph.py's _wrap_node_for_truncation_retry).

    Mirrors /clarify/{run_uuid} almost exactly — same interrupt()/
    Command(resume=...)/MemorySaver checkpoint mechanism, same
    bookkeeping (env overrides, run.json/DB status, background thread
    with its own exception handling) — but resumes with
    {"output_cap": req.output_cap} instead of a free-text answer, and
    the checkpoint resumes execution INSIDE the wrapper around the one
    node that truncated (see _wrap_node_for_truncation_retry), not at
    classify or plan. Only that node re-runs; nothing upstream of it
    does.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    if req.output_cap <= 0:
        raise HTTPException(status_code=400, detail="output_cap must be positive")

    run_dir.mkdir(parents=True, exist_ok=True)

    active_dir = _active_sentinel_dir(run_uuid)
    # Pin this now — everything below in this handler's resume must use
    # THIS directory, not a fresh _active_sentinel_dir() call, for the
    # same reason /clarify pins resume_sentinel_dir: this resume can run
    # for minutes, and a newer chat turn starting mid-resume would make a
    # later fresh lookup silently point at the wrong turn's directory
    # (see /clarify's resume_sentinel_dir comment for the full race).
    retry_sentinel_dir = active_dir
    truncation_file = active_dir / "truncated.json"
    trunc = _read_json(truncation_file) or {}
    stage = trunc.get("stage", "unknown")

    # Remove the sentinel up front, same as /clarify does for
    # clarification.json — the wrapper itself also clears this on
    # resume (belt-and-suspenders; see pipeline/graph.py), but doing it
    # here too means a GET /run between "resume submitted" and "resume
    # thread actually reaches the wrapper's cleanup" doesn't show a
    # stale "waiting_for_truncation_retry" status.
    if truncation_file.exists():
        truncation_file.unlink()

    # Record the retry request itself as a chat message so the
    # conversation shows what happened, mirroring how /clarify records
    # the person's answer as a user turn right away.
    try:
        from storage.chat_store import append_message
        append_message(
            run_uuid, role="user",
            content=f"(Retrying '{stage}' with a higher token limit: {req.output_cap})",
        )
    except Exception as e:
        log.warning("Failed to record retry-truncated request for %s: %s", run_uuid, e)

    env_overrides = _active_runs.get(run_uuid, {}).get("env_overrides", {})
    run_json = _read_json(run_dir / "run.json") or {}

    try:
        from clients.llm import _log_stage_entry
        _log_stage_entry(
            run_dir=str(run_dir), stage=f"{stage}_truncation_retry", model_name="",
            prompt_hash="", tokens_in=0, tokens_out=0, latency_ms=0.0,
            status="ok",
        )
    except Exception as e:
        log.warning("Failed to write truncation_retry marker for %s: %s", run_uuid, e)

    _write_run_json(run_dir, run_uuid, run_json.get("mode"), "running")
    try:
        from storage.critique_store import update_run_status
        update_run_status(run_uuid, "running")
    except Exception:
        pass

    def _resume():
        with _env_lock:
            saved = {k: os.environ.get(k) for k in env_overrides}
            for k, v in env_overrides.items():
                os.environ[k] = v

        try:
            from pipeline.graph import get_graph
            from langgraph.types import Command
            app_graph, callbacks = get_graph()
            config = {"configurable": {"thread_id": run_uuid}}
            if callbacks:
                config["callbacks"] = callbacks

            # Resume value is read by _wrap_node_for_truncation_retry
            # (pipeline/graph.py) as resume_value.get("output_cap") /
            # .get("budget_tokens") — output_cap is the name that
            # actually matches what this is (the max_tokens ceiling),
            # budget_tokens is accepted too since that's the field name
            # PIPELINE_STEP_BUDGET_OVERRIDE already uses elsewhere and a
            # caller might reasonably expect it to work here too.
            final_state = app_graph.invoke(
                Command(resume={"output_cap": req.output_cap}), config=config,
            )

            status = _status_after_invoke(run_uuid, final_state, sentinel_dir=retry_sentinel_dir)

            current = _read_json(run_dir / "run.json") or {}
            if current.get("status") == "cancelled":
                log.info("Run %s was cancelled mid-retry — discarding result", run_uuid)
                return final_state

            _write_run_json(
                run_dir, run_uuid, final_state.get("mode"), status,
                profile=final_state.get("profile"),
            )
            try:
                from storage.critique_store import update_run_status
                update_run_status(run_uuid, status)
            except Exception:
                pass

            if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
                # Truncated again (even at the higher cap), or the
                # retried node itself leads somewhere that halts for
                # clarification — either way, surface it the same way a
                # fresh halt would be.
                _post_halt_chat_message(run_uuid, run_dir, status, sentinel_dir=retry_sentinel_dir)
            else:
                try:
                    from storage.chat_store import append_message
                    reply_text = _extract_reply_text(final_state)
                    append_message(
                        run_uuid, role="assistant", content=reply_text,
                        node_id=None, run_iteration=final_state.get("iteration"),
                    )
                except Exception as e:
                    log.warning("Failed to record post-retry chat turn for %s: %s", run_uuid, e)

            return final_state

        except Exception as exc:
            log.exception("Retry-truncated on run %s failed with an uncaught exception", run_uuid)
            _write_run_json(
                run_dir, run_uuid, run_json.get("mode"), "error",
                error_detail=f"{type(exc).__name__}: {exc}",
            )
            try:
                from storage.critique_store import update_run_status
                update_run_status(run_uuid, "error")
            except Exception:
                pass
            try:
                from storage.chat_store import append_message
                append_message(run_uuid, role="system", content=f"Retry failed: {exc}")
            except Exception:
                pass
            raise

        finally:
            with _env_lock:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    future = _executor.submit(_resume)
    _active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }
    return {"status": "resumed", "stage": stage, "output_cap": req.output_cap}


_ACTIVE_STATUSES = {"running", "waiting_for_clarification", "waiting_for_truncation_retry"}


@app.delete("/run/{run_uuid}")
async def remove_run(run_uuid: str):
    """
    Remove a run — behavior depends on its current state:

      - Still running / waiting for clarification -> CANCEL. The
        background thread finishes its current model call and exits
        (hard mid-inference kill isn't supported — pipeline calls are
        atomic per stage). Status becomes 'cancelled'; the run directory
        and DB row are kept so the user can still see what happened.

      - Already finished (complete/unresolvable/error/cancelled) ->
        DELETE. Removes the run directory from disk and its row from
        the runs table. This is permanent.

      - Directory already missing (e.g. manually cleaned up, or a run
        from before a prior bug) -> treated as already-gone, not an
        error. A 'remove' action on something that's already removed
        should succeed quietly, not 500.
    """
    run_dir = get_run_dir(run_uuid)
    status  = _run_status(run_uuid)

    if status in ("not_found", "unknown") and not run_dir.exists():
        # Nothing to cancel, nothing to delete — but also nothing in the
        # DB necessarily either. Best-effort DB cleanup, then report success.
        try:
            from storage.db import get_conn
            conn = get_conn()
            conn.execute("DELETE FROM runs WHERE run_uuid = ?", (run_uuid,))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return {"status": "already_removed", "run_uuid": run_uuid}

    if status in _ACTIVE_STATUSES:
        run_dir.mkdir(parents=True, exist_ok=True)   # in case it's mid-creation
        _write_run_json(run_dir, run_uuid, None, "cancelled")
        try:
            from storage.critique_store import update_run_status
            update_run_status(run_uuid, "cancelled")
        except Exception:
            pass
        # Do NOT pop _active_runs here. The background thread's future is
        # the only way a later DELETE can tell whether it's actually safe
        # to rmtree the run directory yet — popping it here (the old
        # behaviour) meant a second DELETE had no way to know the thread
        # was still running its current stage, and would rmtree straight
        # out from under it, or the thread's own status write (see the
        # cancellation-race guard in _run_pipeline_thread) would silently
        # clobber "cancelled" back to a terminal status. Cleanup happens
        # in the hard-delete branch below once the future is actually done.
        return {"status": "cancelled", "run_uuid": run_uuid}

    # Finished (complete / unresolvable / error / cancelled) -> hard delete.
    #
    # Cancelled runs are deleted immediately, without waiting on the
    # background thread's future. Previously this branch blocked on
    # fut.done() and 409'd otherwise ("try deleting again in a moment") —
    # reasonable in theory (avoid rmtree-ing out from under a thread still
    # mid-write), but in practice this could 409 indefinitely: a cancelled
    # run's thread is usually just sitting inside one long, uninterruptible
    # LLM call (calls are atomic per stage — see this endpoint's docstring),
    # and that call can legitimately run for the model's full HTTP timeout
    # (routing.yaml http.timeout_seconds, commonly a couple hours) before
    # the future ever becomes done. There was no retry loop on the frontend
    # either, so a user who clicked delete once during that window was left
    # with a run stuck as "cancelled" and permanently undeletable.
    #
    # This is safe to do immediately because "cancelled" is written to
    # run.json synchronously, right above, BEFORE this function returns —
    # so by the time a second DELETE call can even land status is already
    # authoritative on disk. The only remaining risk is the stuck thread
    # eventually waking up and writing INTO a directory we just removed.
    # _run_pipeline_thread and _run_chat_turn_thread already guard against
    # resurrecting a stale status (they check run.json for "cancelled"
    # before writing a terminal status) — but that guard reads run.json,
    # which requires the directory to still exist. _write_run_json's
    # underlying path.write_text would otherwise recreate run_dir with a
    # single orphaned run.json in it. See _write_run_json's own guard
    # (skips the write entirely if run_dir no longer exists) for the other
    # half of this fix — both sides are needed together.
    #
    # complete/unresolvable/error runs still check fut.done() below, since
    # those DON'T get this synchronous status write — a genuinely-still-
    # running thread reaching a terminal status races with this delete for
    # real in that case.
    info = _active_runs.get(run_uuid)
    if status != "cancelled" and info:
        fut = info.get("future")
        if fut and not fut.done():
            raise HTTPException(
                status_code=409,
                detail="This run's current stage is still finishing up — try deleting again in a moment.",
            )
    import shutil
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        pass
    try:
        from storage.db import get_conn
        conn = get_conn()
        conn.execute("DELETE FROM runs WHERE run_uuid = ?", (run_uuid,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("Failed to delete run row for %s: %s", run_uuid, e)

    _active_runs.pop(run_uuid, None)
    return {"status": "deleted", "run_uuid": run_uuid}


# ── SSE stream endpoint ────────────────────────────────────────────────────────

@app.get("/stream/{run_uuid}")
async def stream_run(run_uuid: str):
    """
    Server-Sent Events stream of stages.log entries for a run.
    The client receives one JSON object per completed stage.
    Sends {type: 'complete'}, {type: 'clarification', question: '...'},
    or {type: 'truncated', stage, cap, tokens_out, thinking_block,
    partial_answer} as sentinels.
    """
    async def generator():
        run_dir  = get_run_dir(run_uuid)
        log_path = run_dir / "stages.log"

        # Wait for stages.log to appear. No timeout here on purpose — a
        # resumed run (POST /clarify or /retry-truncated) can legitimately
        # take a while before its first _log_stage_entry() write in
        # llm.py (model load, a slow first call, etc.), and the previous
        # 15s cap closed the connection out from under the client during
        # that gap, which the browser then surfaced as a CORS/connection
        # error rather than a clean timeout. The only real exit condition
        # is the run directory itself going away — that means the run was
        # deleted or never existed, and waiting longer for its log file
        # would never resolve.
        while not log_path.exists():
            if not run_dir.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': 'run not found'})}\n\n"
                return
            # Catches a run that reaches a terminal state (e.g. fails
            # immediately on resume) before ever writing a single
            # stages.log line — otherwise this loop would wait forever
            # for a file that's never coming.
            status = _run_status(run_uuid)
            if status in ("complete", "unresolvable", "error", "cancelled", "interrupted"):
                yield f"data: {json.dumps({'type': 'complete', 'status': status})}\n\n"
                return
            if status == "waiting_for_clarification":
                # Read from _active_sentinel_dir, not a fixed top-level
                # path — the same latent bug fixed for GET /run and
                # _run_status applies here: on a chat follow-up turn,
                # clarify_node writes clarification.json into the TURN
                # directory (run_dir/turns/<seq>/), not run_dir itself.
                clar = _read_json(_active_sentinel_dir(run_uuid) / "clarification.json") or {}
                yield f"data: {json.dumps({'type': 'clarification', **clar})}\n\n"
                return
            if status == "waiting_for_truncation_retry":
                trunc = _read_json(_active_sentinel_dir(run_uuid) / "truncated.json") or {}
                yield f"data: {json.dumps({'type': 'truncated', **trunc})}\n\n"
                return
            await asyncio.sleep(0.5)

        yielded_lines = 0
        while True:
            # Stream new lines from stages.log
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines[yielded_lines:]:
                line = line.strip()
                if line:
                    yield f"data: {line}\n\n"
                    yielded_lines += 1

            # Check for a halt — same _active_sentinel_dir resolution as
            # above and as GET /run, so a mid-stream halt on a chat
            # follow-up turn is found in the right directory.
            active_dir = _active_sentinel_dir(run_uuid)
            clarify_path   = active_dir / "clarification.json"
            truncated_path = active_dir / "truncated.json"
            if clarify_path.exists():
                clar = _read_json(clarify_path) or {}
                yield f"data: {json.dumps({'type': 'clarification', **clar})}\n\n"
                return
            if truncated_path.exists():
                trunc = _read_json(truncated_path) or {}
                yield f"data: {json.dumps({'type': 'truncated', **trunc})}\n\n"
                return

            # Check for completion
            status = _run_status(run_uuid)
            if status in ("complete", "unresolvable", "error", "cancelled", "interrupted"):
                yield f"data: {json.dumps({'type': 'complete', 'status': status})}\n\n"
                return

            await asyncio.sleep(0.8)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":                "no-cache",
            "X-Accel-Buffering":            "no",
            "Access-Control-Allow-Origin":  "*",
        },
    )


# ── Model management endpoints ────────────────────────────────────────────────

@app.get("/models")
async def get_models():
    """List all configured models with health status and assigned roles."""
    from clients.llm import _load_config
    from clients.model_manager import _is_port_alive, current_model

    cfg     = _load_config()
    hot     = current_model()
    roles   = cfg.get("roles", {})

    result = []
    for model_id, mc in cfg.get("models", {}).items():
        port    = mc.get("port")
        alive   = _is_port_alive(port) if port else False
        assigned_roles = [r for r, mid in roles.items() if mid == model_id]
        result.append({
            "id":           model_id,
            "name":         mc.get("name"),
            "quant":        mc.get("quant"),
            "source":       mc.get("source"),
            "port":         port,
            "base_url":     mc.get("base_url"),
            "context_len":  mc.get("context_len"),
            "status":       "loaded" if alive else "unloaded",
            "is_hot":       model_id == hot,
            "roles":        assigned_roles,
            "thinking_default": mc.get("thinking", {}).get("default_on", False),
            "mtp":          mc.get("mtp", {}).get("enabled", False),
        })
    return result


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Manually trigger a model load (swaps out current model)."""
    def _load():
        from clients.model_manager import ensure_model_loaded
        ensure_model_loaded(model_id)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load)
    return {"status": "loaded", "model_id": model_id}


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Stop the currently loaded model server."""
    from clients.model_manager import stop_all, current_model
    if current_model() != model_id:
        raise HTTPException(status_code=400, detail=f"{model_id} is not currently loaded")
    stop_all()
    return {"status": "unloaded", "model_id": model_id}


@app.patch("/models/{model_id}/role/{role}")
async def reassign_role(model_id: str, role: str):
    """Reassign a role to a different model. Writes back to models.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "models.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if model_id not in cfg.get("models", {}):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    if role not in cfg.get("roles", {}):
        raise HTTPException(status_code=404, detail=f"Role {role} not found")

    cfg["roles"][role] = model_id
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Bust config cache
    from clients.llm import _config_cache
    _config_cache.clear()

    return {"status": "updated", "role": role, "model_id": model_id}


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.get("/config/budgets")
async def get_budgets():
    """Return current thinking_budgets from routing.yaml."""
    cfg = _load_routing_config()
    return cfg.get("thinking_budgets", {})


@app.patch("/config/budgets")
async def update_budgets(req: BudgetPatch):
    """Update thinking_budgets in routing.yaml."""
    cfg = _load_routing_config()
    budgets = cfg.setdefault("thinking_budgets", {})
    for stage, tokens in req.budgets.items():
        if tokens < -1:
            raise HTTPException(status_code=400, detail=f"Invalid budget {tokens} for {stage}")
        budgets[stage] = tokens
    _save_routing_config(cfg)
    return {"status": "updated", "budgets": budgets}


@app.post("/config/reload")
async def reload_config():
    """Bust all in-memory config caches. Call after manual YAML edits."""
    from clients.llm import _config_cache, _prompt_cache
    _config_cache.clear()
    _prompt_cache.clear()
    return {"status": "reloaded"}


@app.get("/config/searxng")
async def get_searxng_url():
    """Return current SearXNG URL."""
    return {"url": os.environ.get("SEARXNG_URL", "http://localhost:8888")}


@app.post("/config/searxng")
async def set_searxng_url(body: dict):
    """Update SearXNG URL for this server session."""
    url = body.get("url", "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    os.environ["SEARXNG_URL"] = url
    return {"status": "updated", "url": url}


# ── Lesson endpoints ──────────────────────────────────────────────────────────

@app.get("/lessons")
async def get_lessons(
    task_type:      Optional[str]   = None,
    issue_category: Optional[str]   = None,
    min_confidence: float           = 0.0,
    limit:          int             = 100,
):
    """List lessons with optional filters."""
    return _get_lessons_from_db(task_type, issue_category, min_confidence, limit)


@app.delete("/lessons/{lesson_uuid}")
async def delete_lesson(lesson_uuid: str):
    """Remove a lesson from the store."""
    from storage.db import get_conn
    conn = get_conn()
    try:
        conn.execute("DELETE FROM lessons WHERE lesson_uuid = ?", (lesson_uuid,))
        conn.commit()
        if conn.execute(
            "SELECT changes() as n"
        ).fetchone()["n"] == 0:
            raise HTTPException(status_code=404, detail="Lesson not found")
    finally:
        conn.close()
    return {"status": "deleted", "lesson_uuid": lesson_uuid}


@app.post("/lessons/distill")
async def trigger_distillation():
    """Manually trigger meta-distillation of accumulated lessons."""
    # Placeholder — meta-distillation node not yet implemented
    return {"status": "not_implemented", "message": "Meta-distillation coming soon"}


# ── Chess endpoint ─────────────────────────────────────────────────────────────

@app.post("/chess/analyse")
async def chess_analyse(req: ChessRequest):
    """
    Analyse a chess move. Called by the Bluetooth bridge and Swift apps.
    Returns headline, explanation, suggestion, tacticalPattern.
    fast mode: 512 thinking tokens
    slow mode: 1024 thinking tokens (pass slow_mode=true for blunders/sacrifices)
    """
    run_uuid = str(uuid.uuid4())
    run_dir  = RUNS_DIR / "chess" / run_uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    request_dict = req.model_dump(exclude={"slow_mode"})

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,  # Use default executor (not the pipeline executor)
            lambda: analyse_chess_move(request_dict, str(run_dir), req.slow_mode),
        )
    except Exception as exc:
        # Chess analysis is a synchronous request/response call from the
        # Bluetooth bridge / Swift apps — there's no chat thread or
        # LangGraph checkpoint here to pause and resume the way the main
        # pipeline's retry-truncated flow does (see
        # pipeline/graph.py's _wrap_node_for_truncation_retry), so a
        # TruncatedOutputError here just becomes a clear 502 rather than
        # an opaque unhandled-exception 500. The Swift side can retry the
        # whole request (it already has the position) if it wants to.
        from clients.llm import TruncatedOutputError
        if isinstance(exc, TruncatedOutputError):
            raise HTTPException(
                status_code=502,
                detail=f"Chess analysis truncated at max_tokens={exc.cap} "
                       f"({exc.tokens_out} tokens generated) — try again, "
                       f"or use fast mode if this was slow mode.",
            )
        raise
    return result


# ── Custom pipeline CRUD ────────────────────────────────────────────────────
# Create/list/edit/delete/validate custom pipeline definitions
# (config/pipelines/{name}.yaml). Every write goes through the same
# validate_pipeline_definition() the compiler itself uses (pipeline/
# custom_validator.py), so a definition that fails to save would also
# have failed to compile — the GUI never gets a false "saved OK" that
# then blows up at run time.

def _wire_to_definition(body: PipelineDefIn) -> "PipelineDefinition":
    """Convert the API's flat wire format into the real tagged-union schema."""
    from schemas.pipeline_def import (
        PipelineDefinition, ExistingStep, FreeformStep, DecisionStep,
        DecisionOutcome, StepType,
    )

    steps = []
    for s in body.steps:
        if s.type == "existing":
            if not s.node_name:
                raise HTTPException(400, f"Step '{s.id}': type=existing requires node_name")
            steps.append(ExistingStep(
                id=s.id, node_name=s.node_name,
                model_override=s.model_override, budget_override=s.budget_override,
            ))
        elif s.type == "freeform":
            if not s.model or not s.system_prompt or not s.output_key:
                raise HTTPException(
                    400, f"Step '{s.id}': type=freeform requires model, system_prompt, output_key"
                )
            steps.append(FreeformStep(
                id=s.id, model=s.model,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                thinking=s.thinking or False,
                system_prompt=s.system_prompt,
                user_template=s.user_template or "{input}",
                input_key=s.input_key or "normalised_input",
                output_key=s.output_key,
                feedback_mode=s.feedback_mode or "auto",
            ))
        elif s.type == "decision":
            if not s.system_prompt or not s.outcomes:
                raise HTTPException(
                    400, f"Step '{s.id}': type=decision requires system_prompt, outcomes"
                )
            steps.append(DecisionStep(
                id=s.id, model=s.model or "9b",
                thinking=s.thinking or False,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                system_prompt=s.system_prompt,
                input_key=s.input_key or "normalised_input",
                outcomes=[DecisionOutcome(**o) for o in s.outcomes],
                is_loop_back=s.is_loop_back or False,
                max_iterations=s.max_iterations,
                feedback_mode=s.feedback_mode or "auto",
            ))
        else:
            raise HTTPException(400, f"Step '{s.id}': unknown type '{s.type}'")

    return PipelineDefinition(
        name=body.name, description=body.description,
        entry_step=body.entry_step, steps=steps,
        edge_overrides=body.edge_overrides,
        max_total_iterations=body.max_total_iterations,
    )


@app.get("/pipelines")
async def list_pipelines():
    """List all saved custom pipeline definitions (name + description only)."""
    from pipeline.custom_graph import PIPELINES_DIR, load_pipeline_definition

    result = []
    for path in sorted(PIPELINES_DIR.glob("*.yaml")):
        try:
            defn = load_pipeline_definition(path.stem)
            result.append({
                "name": defn.name, "description": defn.description,
                "step_count": len(defn.steps), "entry_step": defn.entry_step,
            })
        except Exception as e:
            result.append({"name": path.stem, "description": f"[INVALID: {e}]", "step_count": 0})
    return result


@app.get("/pipelines/{name}")
async def get_pipeline(name: str):
    """Full definition for one custom pipeline — the shape a GUI editor needs."""
    from pipeline.custom_graph import load_pipeline_definition
    try:
        defn = load_pipeline_definition(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    return json.loads(defn.model_dump_json())


@app.post("/pipelines/validate")
async def validate_pipeline(body: PipelineDefIn):
    """
    Validate a pipeline definition WITHOUT saving it. Lets a GUI show
    live errors while the user is still editing, before they commit.
    """
    from pipeline.custom_validator import validate_pipeline_definition
    try:
        defn = _wire_to_definition(body)   # raises HTTPException on structural issues
    except HTTPException:
        raise
    except Exception as e:
        # Pydantic validation errors (duplicate ids, missing max_iterations
        # on a self-declared loop-back, bad entry_step, etc.)
        return {"valid": False, "errors": [str(e)]}

    errors = validate_pipeline_definition(defn)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/pipelines")
async def create_pipeline(body: PipelineDefIn):
    """
    Create or overwrite a custom pipeline definition. Validates before
    writing — a pipeline that fails validation is never saved to disk,
    so config/pipelines/ never contains a definition that would fail to
    compile at run time.
    """
    from pipeline.custom_graph import PIPELINES_DIR
    from pipeline.custom_validator import validate_pipeline_definition

    defn = _wire_to_definition(body)
    errors = validate_pipeline_definition(defn)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Pipeline failed validation, not saved", "errors": errors},
        )

    path = PIPELINES_DIR / f"{body.name}.yaml"
    import yaml
    with open(path, "w") as f:
        yaml.dump(json.loads(defn.model_dump_json()), f, default_flow_style=False, sort_keys=False)

    return {"status": "saved", "name": body.name, "path": str(path)}


@app.delete("/pipelines/{name}")
async def delete_pipeline(name: str):
    """Delete a custom pipeline definition. Does not affect past runs."""
    from pipeline.custom_graph import PIPELINES_DIR
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    path.unlink()
    return {"status": "deleted", "name": name}


@app.get("/pipelines/nodes/available")
async def list_available_node_types():
    """
    What a GUI needs to populate an 'add step' dropdown: every reusable
    existing-node name, and the step-type shapes (freeform/decision)
    with their required fields, sourced directly from the real schema
    so this never drifts out of sync with what the backend actually accepts.
    """
    from schemas.pipeline_def import EXISTING_NODE_NAMES

    return {
        "existing_nodes": EXISTING_NODE_NAMES,
        "step_types": {
            "existing": {
                "required": ["id", "node_name"],
                "optional": ["model_override", "budget_override"],
            },
            "freeform": {
                "required": ["id", "model", "system_prompt", "output_key"],
                "optional": ["budget_tokens", "thinking", "user_template",
                             "input_key", "feedback_mode"],
            },
            "decision": {
                "required": ["id", "system_prompt", "outcomes"],
                "optional": ["model", "thinking", "budget_tokens", "input_key",
                             "is_loop_back", "max_iterations", "feedback_mode"],
                "outcome_shape": {"value": "string", "next_step": "string (step id or __end__)",
                                   "description": "string, optional"},
            },
        },
    }


# ── Health endpoint ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Server health, active model, and queue status."""
    from clients.model_manager import current_model, _is_port_alive
    from clients.llm import _load_config

    cfg    = _load_config()
    hot    = current_model()
    hot_cfg = cfg["models"].get(hot, {}) if hot else {}

    active_count = sum(
        1 for r in _active_runs.values()
        if (f := r.get("future")) and f.running()
    )

    return {
        "status":       "ok",
        "hot_model":    hot,
        "hot_port":     hot_cfg.get("port"),
        "hot_context":  hot_cfg.get("context_len"),
        "active_runs":  active_count,
        "queue_depth":  _executor._work_queue.qsize(),
        "searxng_url":  os.environ.get("SEARXNG_URL", "http://localhost:8888"),
    }


# ── Static / UI ───────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not built. Place index.html in api/static/"}
    return FileResponse(str(index))


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    from storage.db import initialise
    initialise()
    log.info("Eolophus API server started. UI: http://localhost:8000")