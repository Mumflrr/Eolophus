"""
api/routers/runs.py — run creation, retrieval, listing, deletion, the
uploaded image, and attachment management.

_run_pipeline_thread is the background worker for a freshly-created run
(as opposed to a chat follow-up — see routers/chat.py — or a resume —
see routers/clarify.py, which share the same halt/status machinery from
api/status.py but re-enter the graph differently).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from pipeline.state import PipelineState
import json
import logging
import os
import uuid
from pathlib import Path
from typing import cast, Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api import state
from api.attachments import (
    compose_input_with_attachments,
    ext_to_mime_map,
    load_live_attachments,
    update_attachment_manifest,
    write_attachments,
    write_image,
)
from api.json_utils import read_json, write_run_json
from api.paths import RUNS_DIR, active_sentinel_dir, get_run_dir
from api.schemas import AddAttachmentsIn, ImageIn, RunRequest
from api.status import extract_reply_text, run_status, status_after_invoke

log = logging.getLogger(__name__)

router = APIRouter()

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
        data = read_json(dir_path / fname)
        if data is not None:
            artifacts[fname.replace(".json", "")] = data
    return artifacts


def _read_iteration_snapshots(dir_path: Path) -> dict:
    """
    Read run_dir/iterations/<n>/*.json (or turn_dir/iterations/<n>/*.json
    for a chat follow-up). Returns {"0": {"fixed": {...}, "verdict": {...}},
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
                   last run (edits made via the CRUD endpoints take effect
                   on the NEXT run with no server restart needed).
    """
    run_dir = Path(initial_state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with state.env_lock:
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

        config: RunnableConfig = {"configurable": {"thread_id": run_uuid}}
        if callbacks:
            config["callbacks"] = callbacks

        write_run_json(run_dir, run_uuid, initial_state.get("mode"), "running")
        final_state = app_graph.invoke(cast(PipelineState, initial_state), config=config)

        # A node that called interrupt() (clarify_node, or the generic
        # truncation-retry wrapper) makes invoke() return early with an
        # "__interrupt__" entry instead of running to completion. See
        # status_after_invoke's docstring for how clarify vs truncation
        # halts are told apart.
        status = status_after_invoke(run_uuid, final_state)

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
        current = read_json(run_dir / "run.json") or {}
        if current.get("status") == "cancelled":
            log.info("Run %s was cancelled mid-flight — discarding pipeline result", run_uuid)
            return final_state

        write_run_json(
            run_dir, run_uuid, initial_state.get("mode"), status,
            profile=final_state.get("profile"),
        )
        update_run_status(run_uuid, status)

        if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
            from api.status import post_halt_chat_message
            post_halt_chat_message(run_uuid, run_dir, status)
        else:
            # Mirror chat._run_chat_turn_thread: persist the assistant's
            # reply as a chat turn once the run reaches a terminal state.
            try:
                from storage.chat_store import append_message
                reply_text = extract_reply_text(final_state)
                append_message(
                    run_uuid, role="assistant", content=reply_text,
                    node_id=None, run_iteration=final_state.get("iteration"),
                )
            except Exception as e:
                log.warning("Failed to record initial-run assistant reply for %s: %s", run_uuid, e)

        return final_state

    except Exception as exc:
        log.exception("Pipeline run %s failed with an uncaught exception", run_uuid)
        write_run_json(
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
        with state.env_lock:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


@router.get("/debug/run/{run_uuid}/exception")
async def debug_run_exception(run_uuid: str):
    info = state.active_runs.get(run_uuid)
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


@router.get("/run/{run_uuid}/image")
async def get_run_image(run_uuid: str):
    """
    Serve the run's uploaded image bytes (see write_image / ImageIn).
    Not served via the /static mount since RUNS_DIR isn't under it —
    this is a plain FileResponse instead, with the mime type re-derived
    from the extension write_image chose.
    """
    run_dir = get_run_dir(run_uuid)
    matches = list(run_dir.glob("image.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="This run has no image")

    path = matches[0]
    media_type = ext_to_mime_map().get(path.suffix.lstrip("."), "application/octet-stream")
    return FileResponse(path, media_type=media_type)


@router.post("/run")
async def start_run(req: RunRequest):
    """Start a new pipeline run. Returns immediately with run_uuid."""
    run_uuid = str(uuid.uuid4())
    run_dir  = get_run_dir(run_uuid)
    run_dir.mkdir(parents=True, exist_ok=True)

    attachment_manifest = write_attachments(run_dir, req.attachments)
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
        raw_image_path = write_image(run_dir, req.image)

    composed_input = compose_input_with_attachments(
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
    # (hence the env_lock around concurrent runs stomping on each other) —
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
    write_run_json(run_dir, run_uuid, None, "running")

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

    future = state.executor.submit(
        _run_pipeline_thread, run_uuid, initial_state, env_overrides, req.pipeline
    )
    # env_overrides is stored here so a later POST /clarify on this run can
    # re-apply whatever env vars this run started with when it resumes —
    # see clarify.clarify_run. Previously this dict never made it into
    # _active_runs, so clarify_run's lookup always saw {}.
    state.active_runs[run_uuid] = {
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


@router.get("/runs")
async def list_runs(limit: int = 50):
    """List recent pipeline runs from the database."""
    from api.db_helpers import get_runs_from_db

    db_runs = get_runs_from_db(limit)
    # Enrich with live status derived from disk, not just the DB row.
    # Previously this only ran for uid in state.active_runs — which is
    # populated in memory and is EMPTY right after a server restart, even
    # though the run's actual state (running/halted/cancelled/etc.) still
    # lives on disk via run.json and the sentinel files. A row whose
    # background thread died with the old process kept showing whatever
    # status was last written before the restart (often "running"),
    # forever, since nothing here ever re-checked it. run_status() is a
    # handful of small file reads per run; recomputing it for every listed
    # row (not just in-memory-active ones) keeps the list honest across
    # restarts at negligible cost for the list sizes this endpoint serves.
    for r in db_runs:
        uid = r.get("run_uuid")
        if uid:
            r["status"] = run_status(uid)
    return db_runs


@router.get("/run/{run_uuid}")
async def get_run(run_uuid: str):
    """Full run detail: artifacts, stages log, current status, each chat
    turn's own artifact snapshot, and each correction-loop iteration's own
    snapshot of fixed/verdict/draft/critique."""
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
    # Every chat turn writes into its own run_dir/turns/<seq>/ instead of
    # overwriting the top-level files above, so this is what lets the UI
    # show e.g. "classify.json for message 3" distinctly from message 1's
    # classify.json. Each turn can ALSO have its own correction loop, so
    # turn_iterations mirrors the top-level iterations map but scoped per
    # turn seq: {seq: {iteration: {...}}}.
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
    # chat follow-up is in flight (see active_sentinel_dir's docstring for
    # why this can't just be run_dir unconditionally: clarify_node and the
    # truncation-retry wrapper both write via Path(state["run_dir"]),
    # which IS the turn dir for a chat follow-up).
    active_dir    = active_sentinel_dir(run_uuid)
    clarification = read_json(active_dir / "clarification.json")
    truncation    = read_json(active_dir / "truncated.json")

    # Surface the uploaded image (if any) as a relative path the frontend
    # can build a static URL from — same pattern as attachments.json's
    # "path" field. Doesn't try every extension exhaustively via glob
    # since write_image only ever writes exactly one image.* file per run.
    image_path = None
    matches = list(run_dir.glob("image.*"))
    if matches:
        image_path = str(matches[0].relative_to(run_dir))

    return {
        "run_uuid":        run_uuid,
        "status":          run_status(run_uuid),
        "artifacts":       artifacts,
        "iterations":      iterations,
        "chat_artifacts":  chat_artifacts,
        "turn_iterations": turn_iterations,
        "image_path":      image_path,
        "stages":          stages,
        "clarification":   clarification,
        "truncation":      truncation,
    }


@router.post("/run/{run_uuid}/attachments")
async def add_attachments(run_uuid: str, req: AddAttachmentsIn):
    """
    Add one or more new text/code attachments to an already-started run,
    for use from an ongoing chat (runDetail.js) rather than only at
    creation time (runs.js's POST /run).

    Reuses write_attachments for on-disk persistence (same de-dupe-by-
    basename behavior as the initial run), then merges the new entries
    into the existing attachments.json instead of overwriting it, so
    attachments added at run-start and mid-chat live in one manifest.
    Newly-added files are picked up on the *next* turn via
    load_live_attachments — same mechanism as everything else in that
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

    new_entries = write_attachments(run_dir, req.attachments)

    manifest_path = run_dir / "attachments.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        manifest = []
    manifest.extend(new_entries)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"status": "added", "attachments": new_entries}


@router.delete("/run/{run_uuid}/attachments/{filename}")
async def exclude_attachment(run_uuid: str, filename: str):
    """
    Exclude an attachment from future turns without deleting it. Once
    excluded, load_live_attachments skips it, so it stops being folded
    into normalised_input (and stops costing tokens) on the run's current
    turn and every turn after — but the file and its manifest entry are
    kept, so it still shows up in the run's artifacts view and can be
    brought back with POST .../attachments/{filename}/include.
    """
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    entry = update_attachment_manifest(run_dir, filename, excluded=True)
    return {"status": "excluded", "attachment": entry}


@router.post("/run/{run_uuid}/attachments/{filename}/include")
async def include_attachment(run_uuid: str, filename: str):
    """Re-include a previously excluded attachment — the inverse of
    DELETE .../attachments/{filename}. No re-upload needed; the file was
    never removed from disk, only skipped when folding input."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    entry = update_attachment_manifest(run_dir, filename, excluded=False)
    return {"status": "included", "attachment": entry}


@router.delete("/run/{run_uuid}")
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
    status  = run_status(run_uuid)

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

    if status in state.ACTIVE_STATUSES:
        run_dir.mkdir(parents=True, exist_ok=True)   # in case it's mid-creation
        write_run_json(run_dir, run_uuid, None, "cancelled")
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
        #
        # Clear every halt sentinel under this run — top-level AND every
        # turns/<n>/ dir, not just active_sentinel_dir()'s current pick.
        # run_status() checks clarification.json/truncated.json BEFORE it
        # ever reads run.json's status (see status.py), so leaving one in
        # place after writing "cancelled" here means the very next
        # run_status() call — including the one at the top of a follow-up
        # DELETE — still reports "waiting_for_clarification" or
        # "waiting_for_truncation_retry" and lands right back in this same
        # branch. That made a halted run permanently undeletable through
        # the API: cancel writes "cancelled", the sentinel silently
        # outranks it forever, and the hard-delete branch below never
        # becomes reachable — the only way out was deleting the sentinel
        # file by hand. A cancelled run will never be resumed, so nuking
        # every sentinel here (not just the currently-active one) is safe.
        for sentinel_name in ("clarification.json", "truncated.json"):
            for sentinel_path in run_dir.rglob(sentinel_name):
                try:
                    sentinel_path.unlink()
                except OSError:
                    pass
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
    # _run_pipeline_thread and chat._run_chat_turn_thread already guard
    # against resurrecting a stale status (they check run.json for
    # "cancelled" before writing a terminal status) — but that guard reads
    # run.json, which requires the directory to still exist.
    # write_run_json's underlying path.write_text would otherwise recreate
    # run_dir with a single orphaned run.json in it. See write_run_json's
    # own guard (skips the write entirely if run_dir no longer exists) for
    # the other half of this fix — both sides are needed together.
    #
    # complete/unresolvable/error runs still check fut.done() below, since
    # those DON'T get this synchronous status write — a genuinely-still-
    # running thread reaching a terminal status races with this delete for
    # real in that case.
    info = state.active_runs.get(run_uuid)
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

    state.active_runs.pop(run_uuid, None)
    return {"status": "deleted", "run_uuid": run_uuid}