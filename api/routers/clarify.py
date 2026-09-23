"""
api/routers/clarify.py — resuming a halted run.

Two halt kinds share this module because they share almost identical
resume bookkeeping (pin the sentinel dir, apply env_overrides, write
run.json/DB status, background thread with its own exception handling)
and both drive the same interrupt()/Command(resume=...)/MemorySaver
checkpoint mechanism — they just differ in resume payload and in which
node the checkpoint resumes inside.
"""

from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig

from fastapi import APIRouter, HTTPException

from api import state
from clients.llm import RunCancelled, begin_run, clear_cancel, end_run
from api.json_utils import read_json, write_run_json
from api.paths import active_sentinel_dir, get_run_dir
from api.schemas import ClarifyRequest, RetryTruncatedRequest
from api.status import extract_reply_text, post_halt_chat_message, status_after_invoke

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/clarify/{run_uuid}")
async def clarify_run(run_uuid: str, req: ClarifyRequest):
    """
    Resume a pipeline that has halted for clarification.

    Mirrors runs._run_pipeline_thread's bookkeeping, which this endpoint
    previously skipped entirely:
      - run_dir.mkdir(): defensive, same as _run_pipeline_thread — avoids
        a FileNotFoundError from llm.py's log writers if the directory
        was ever missing at resume time.
      - env_overrides applied under state.env_lock: whatever env vars
        this run was started with were previously lost on resume, since
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
    # resume thread below. active_sentinel_dir() always resolves to
    # whatever the LATEST turn directory is *at the moment it's called* —
    # it has no notion of "which resume this call belongs to". The actual
    # app_graph.invoke(Command(resume=...)) call below can run for minutes
    # (bugfix/audit/validate are slow stages), and if a NEW chat turn
    # starts anywhere in that window — another message, another halt, any
    # code path that creates run_dir/turns/<n+1>/ — then a later call to
    # active_sentinel_dir() (e.g. from a GET /run poll, or from
    # status_after_invoke() below) will resolve to that newer turn's
    # directory instead of the one THIS resume is actually writing into.
    # That mismatch is what caused a run to look stuck on
    # "waiting_for_clarification" even after the person answered and the
    # graph had genuinely finished: the finished run's real output landed
    # in the turn directory captured here, while status-checking code
    # was — by the time anyone polled it — looking at a different,
    # unrelated turn directory that active_sentinel_dir() now considered
    # "latest". Resolve once, pass the pinned path everywhere this
    # specific resume needs to check or clear a sentinel, and don't
    # re-derive "latest" partway through handling one resume.
    resume_sentinel_dir = active_sentinel_dir(run_uuid)

    # Remove clarification sentinel from that pinned directory — NOT a
    # fresh active_sentinel_dir() call, for the reason above.
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

    env_overrides = state.active_runs.get(run_uuid, {}).get("env_overrides", {})
    run_json = read_json(run_dir / "run.json") or {}

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

    write_run_json(run_dir, run_uuid, run_json.get("mode"), "running")
    try:
        from storage.critique_store import update_run_status
        update_run_status(run_uuid, "running")
    except Exception:
        pass

    def _resume():
        with state.env_lock:
            saved = {k: os.environ.get(k) for k in env_overrides}
            for k, v in env_overrides.items():
                os.environ[k] = v

        try:
            begin_run(run_uuid)   # raises RunCancelled if cancelled while still queued
            from pipeline.graph import get_graph
            from langgraph.types import Command
            app_graph, callbacks = get_graph()
            config: RunnableConfig = {"configurable": {"thread_id": run_uuid}}
            if callbacks:
                config["callbacks"] = callbacks

            final_state = app_graph.invoke(Command(resume=req.answer), config=config)

            # Same "halt before pipeline_failed" ordering as
            # runs._run_pipeline_thread — a second halt (clarify or
            # truncation) on this resume must not be misreported as
            # unresolvable/complete. Pass the pinned resume_sentinel_dir
            # captured before this background thread started — see its
            # definition above for why re-deriving "latest" here could
            # silently point at an unrelated, newer chat turn's directory.
            status = status_after_invoke(run_uuid, final_state, sentinel_dir=resume_sentinel_dir)

            # Same cancellation-race guard as runs._run_pipeline_thread:
            # the run could have been cancelled while this resume was
            # mid-flight.
            current = read_json(run_dir / "run.json") or {}
            if current.get("status") == "cancelled":
                log.info("Run %s was cancelled mid-resume — discarding result", run_uuid)
                return final_state

            write_run_json(
                run_dir, run_uuid, final_state.get("mode"), status,
                profile=final_state.get("profile"),
            )
            try:
                from storage.critique_store import update_run_status
                update_run_status(run_uuid, status)
            except Exception:
                pass

            # Same append_message pattern as runs._run_pipeline_thread: a
            # fresh halt gets its question/truncation details posted as an
            # assistant turn; anything terminal gets the actual reply.
            if status in ("waiting_for_clarification", "waiting_for_truncation_retry"):
                post_halt_chat_message(run_uuid, run_dir, status, sentinel_dir=resume_sentinel_dir)
            else:
                try:
                    from storage.chat_store import append_message
                    reply_text = extract_reply_text(final_state)
                    append_message(
                        run_uuid, role="assistant", content=reply_text,
                        node_id=None, run_iteration=final_state.get("iteration"),
                    )
                except Exception as e:
                    log.warning("Failed to record post-resume chat turn for %s: %s", run_uuid, e)

            return final_state

        except RunCancelled:
            # DELETE /run/{uuid} already wrote status="cancelled"; nothing to overwrite.
            log.info("Resume on run %s cancelled — worker thread exited early", run_uuid)
            return {}

        except Exception as exc:
            log.exception("Resume (clarify) on run %s failed with an uncaught exception", run_uuid)
            write_run_json(
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
            end_run()
            with state.env_lock:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    clear_cancel(run_uuid)   # a previous cancel of this run must not abort the resume
    future = state.executor.submit(_resume)
    state.active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }
    return {"status": "resumed"}


@router.post("/run/{run_uuid}/retry-truncated")
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

    active_dir = active_sentinel_dir(run_uuid)
    # Pin this now — everything below in this handler's resume must use
    # THIS directory, not a fresh active_sentinel_dir() call, for the
    # same reason /clarify pins resume_sentinel_dir: this resume can run
    # for minutes, and a newer chat turn starting mid-resume would make a
    # later fresh lookup silently point at the wrong turn's directory
    # (see clarify_run's resume_sentinel_dir comment for the full race).
    retry_sentinel_dir = active_dir
    truncation_file = active_dir / "truncated.json"
    trunc = read_json(truncation_file) or {}
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

    env_overrides = state.active_runs.get(run_uuid, {}).get("env_overrides", {})
    run_json = read_json(run_dir / "run.json") or {}

    try:
        from clients.llm import _log_stage_entry
        _log_stage_entry(
            run_dir=str(run_dir), stage=f"{stage}_truncation_retry", model_name="",
            prompt_hash="", tokens_in=0, tokens_out=0, latency_ms=0.0,
            status="ok",
        )
    except Exception as e:
        log.warning("Failed to write truncation_retry marker for %s: %s", run_uuid, e)

    write_run_json(run_dir, run_uuid, run_json.get("mode"), "running")
    try:
        from storage.critique_store import update_run_status
        update_run_status(run_uuid, "running")
    except Exception:
        pass

    def _resume():
        with state.env_lock:
            saved = {k: os.environ.get(k) for k in env_overrides}
            for k, v in env_overrides.items():
                os.environ[k] = v

        try:
            begin_run(run_uuid)   # raises RunCancelled if cancelled while still queued
            from pipeline.graph import get_graph
            from langgraph.types import Command
            app_graph, callbacks = get_graph()
            config: RunnableConfig = {"configurable": {"thread_id": run_uuid}}
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

            status = status_after_invoke(run_uuid, final_state, sentinel_dir=retry_sentinel_dir)

            current = read_json(run_dir / "run.json") or {}
            if current.get("status") == "cancelled":
                log.info("Run %s was cancelled mid-retry — discarding result", run_uuid)
                return final_state

            write_run_json(
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
                post_halt_chat_message(run_uuid, run_dir, status, sentinel_dir=retry_sentinel_dir)
            else:
                try:
                    from storage.chat_store import append_message
                    reply_text = extract_reply_text(final_state)
                    append_message(
                        run_uuid, role="assistant", content=reply_text,
                        node_id=None, run_iteration=final_state.get("iteration"),
                    )
                except Exception as e:
                    log.warning("Failed to record post-retry chat turn for %s: %s", run_uuid, e)

            return final_state

        except RunCancelled:
            # DELETE /run/{uuid} already wrote status="cancelled"; nothing to overwrite.
            log.info("Resume on run %s cancelled — worker thread exited early", run_uuid)
            return {}

        except Exception as exc:
            log.exception("Retry-truncated on run %s failed with an uncaught exception", run_uuid)
            write_run_json(
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
            end_run()
            with state.env_lock:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    clear_cancel(run_uuid)   # a previous cancel of this run must not abort the resume
    future = state.executor.submit(_resume)
    state.active_runs[run_uuid] = {
        "future":        future,
        "run_dir":       str(run_dir),
        "env_overrides": env_overrides,
    }
    return {"status": "resumed", "stage": stage, "output_cap": req.output_cap}