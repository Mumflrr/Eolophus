"""
api/routers/stream.py — SSE stream of stages.log entries for a run.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.json_utils import read_json
from api.paths import active_sentinel_dir, get_run_dir
from api.status import clarification_already_answered, run_status

router = APIRouter()


@router.get("/stream/{run_uuid}")
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
            status = run_status(run_uuid)
            if status in ("complete", "unresolvable", "error", "cancelled", "interrupted"):
                yield f"data: {json.dumps({'type': 'complete', 'status': status})}\n\n"
                return
            if status == "waiting_for_clarification":
                # Read from active_sentinel_dir, not a fixed top-level
                # path — the same latent bug fixed for GET /run and
                # run_status applies here: on a chat follow-up turn,
                # clarify_node writes clarification.json into the TURN
                # directory (run_dir/turns/<seq>/), not run_dir itself.
                sentinel_path = active_sentinel_dir(run_uuid) / "clarification.json"
                # Cross-check against the chat log before trusting the
                # sentinel file — see clarification_already_answered()'s
                # docstring for why active_sentinel_dir() can (briefly,
                # harmlessly under normal timing) resolve to the wrong
                # turn directory right after a chat submit, and why the
                # chat log's ordering is the one signal that race can't
                # corrupt. If the file is somehow still here despite the
                # chat log showing it answered, that's now a confirmed
                # stale leftover rather than an ambiguous read — delete
                # it so nothing downstream (a future reconnect, a GET
                # /run poll) gets fooled by it again, and fall through to
                # keep waiting on the run's actual current state instead
                # of reporting a halt that's already resolved.
                if clarification_already_answered(run_uuid):
                    try:
                        sentinel_path.unlink()
                    except OSError:
                        pass
                else:
                    clar = read_json(sentinel_path) or {}
                    yield f"data: {json.dumps({'type': 'clarification', **clar})}\n\n"
                    return
            if status == "waiting_for_truncation_retry":
                trunc = read_json(active_sentinel_dir(run_uuid) / "truncated.json") or {}
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

            # Check for a halt — same active_sentinel_dir resolution as
            # above and as GET /run, so a mid-stream halt on a chat
            # follow-up turn is found in the right directory.
            active_dir = active_sentinel_dir(run_uuid)
            clarify_path   = active_dir / "clarification.json"
            truncated_path = active_dir / "truncated.json"
            if clarify_path.exists():
                # Same cross-check as the pre-log-file branch above — see
                # clarification_already_answered()'s docstring. This is
                # the path that actually reproduced the reported bug:
                # openStream() reconnects after every chat submit, this
                # loop's very first iteration replays all of stages.log,
                # and immediately after that replay it re-checks
                # active_sentinel_dir() fresh — which can still be
                # pointing at the PREVIOUS turn if the new turn's
                # directory hasn't been created by the background thread
                # yet. Trusting file-existence alone there is exactly
                # what let an already-answered clarification reappear on
                # a later, unrelated chat turn.
                if clarification_already_answered(run_uuid):
                    try:
                        clarify_path.unlink()
                    except OSError:
                        pass
                else:
                    clar = read_json(clarify_path) or {}
                    yield f"data: {json.dumps({'type': 'clarification', **clar})}\n\n"
                    return
            if truncated_path.exists():
                trunc = read_json(truncated_path) or {}
                yield f"data: {json.dumps({'type': 'truncated', **trunc})}\n\n"
                return

            # Check for completion
            status = run_status(run_uuid)
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