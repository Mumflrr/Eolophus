"""
api/status.py — deriving run status from on-disk sentinels, and posting
halt (clarification / truncation) events as chat messages.

This logic is shared verbatim by runs.py, chat.py, clarify.py, and
stream.py — every one of them needs to answer "is this run currently
halted, and if so which kind of halt" the same way, resolved against
the same turn directory. Keeping it in one module is what makes the
turn-directory fix (see api.paths.active_sentinel_dir) apply
consistently: a router that reimplemented this would silently drift
back into the old top-level-only bug.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from api.json_utils import read_json
from api.paths import active_sentinel_dir, get_run_dir
from api import state

log = logging.getLogger(__name__)


def status_after_invoke(run_uuid: str, final_state: dict, sentinel_dir: Optional[Path] = None) -> str:
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
    resolution run_status() uses for a later GET, kept consistent here so
    a status computed right after invoke() matches what a follow-up GET
    would report.

    sentinel_dir: pass the directory THIS invocation is actually writing
    into when the caller already knows it (e.g. clarify.py's resume
    path, which pins it before starting the background resume — see
    that endpoint's comment for why re-deriving "latest" here would be
    wrong if a newer chat turn started while this invocation was
    running). Falls back to a fresh active_sentinel_dir() lookup for
    callers that don't have a pinned directory (e.g. the original,
    non-resumed runs._run_pipeline_thread, where "latest turn" and "this
    run" are always the same directory since no follow-up turn exists
    yet).
    """
    if not final_state.get("__interrupt__"):
        return "unresolvable" if final_state.get("pipeline_failed") else "complete"
    active_dir = sentinel_dir if sentinel_dir is not None else active_sentinel_dir(run_uuid)
    if (active_dir / "truncated.json").exists():
        return "waiting_for_truncation_retry"
    return "waiting_for_clarification"


def post_halt_chat_message(run_uuid: str, run_dir: Path, status: str, sentinel_dir: Optional[Path] = None) -> None:
    """
    Surface a fresh halt (clarification or truncation) as an assistant
    chat message, same as runs._run_pipeline_thread has always done for
    clarification. For truncation, node_id="truncated" lets runDetail.js
    render a distinct "Retry with higher limit" affordance instead of the
    clarify answer box — see the truncation payload's stage/cap/
    thinking_block/partial_answer fields (pipeline/graph.py's
    _wrap_node_for_truncation_retry), which flow through here into the
    posted message's content and metadata_json so retryTruncated's UI
    has something concrete to show and act on.

    sentinel_dir: pass the directory the triggering invocation actually
    wrote its sentinel into, when the caller already knows it (see
    status_after_invoke's docstring for why re-deriving "latest" via a
    fresh active_sentinel_dir() call can pick the wrong directory if a
    newer chat turn started while a resume was still in flight).
    """
    from storage.chat_store import append_message
    active_dir = sentinel_dir if sentinel_dir is not None else active_sentinel_dir(run_uuid)
    try:
        if status == "waiting_for_clarification":
            clar = read_json(active_dir / "clarification.json") or {}
            question = clar.get("question", "Clarification needed.")
            append_message(run_uuid, role="assistant", content=question, node_id="clarify")
        elif status == "waiting_for_truncation_retry":
            trunc = read_json(active_dir / "truncated.json") or {}
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


def clarification_already_answered(run_uuid: str) -> bool:
    """
    True if the chat log's own ordering already shows this clarification
    resolved, independent of whether a clarification.json sentinel file
    still happens to exist anywhere on disk.

    Every other check in this codebase (run_status, stream.py) answers
    "is there a pending clarification" by asking "does a sentinel file
    exist at whatever path active_sentinel_dir() resolves to right now."
    That path is racy: active_sentinel_dir()/latest_turn_seq() pick
    "latest" by scanning which turns/<n>/ directories currently exist on
    disk, but chat._run_chat_turn_thread only mkdir()s its own turn
    directory once the executor's single worker thread actually starts
    running it — not when post_chat_message returns. In the window
    between "client got its 200 and reconnected the stream" and "the
    background thread created its directory," active_sentinel_dir()
    still (correctly, at that instant) reports the PREVIOUS turn as
    latest. That's harmless as long as the previous turn's sentinel was
    actually deleted, but nothing can tell the difference between
    "genuinely gone" and "briefly looking at the wrong directory" from
    file existence alone — there's no notion of identity, just "found a
    file or didn't."

    The chat table has no such ambiguity: clarify_run appends the
    person's answer as a 'user' row SYNCHRONOUSLY, before it ever
    returns the HTTP response (see clarify.py) — well before any
    reconnect or poll can happen. So "is the most recent turn an
    unanswered clarify halt" is a fact about message ordering that
    doesn't depend on which directory any sentinel file lives in, and
    can't be raced by a lazy mkdir() the way file-existence checks can.
    """
    from storage.chat_store import get_messages
    msgs = get_messages(run_uuid)
    if not msgs:
        return False
    last = msgs[-1]
    return not (last["role"] == "assistant" and last.get("node_id") == "clarify")


def run_status(run_uuid: str) -> str:
    """
    Derive run status from sentinel files in the run directory.

    Checks the LATEST turn directory first (the common case — an active
    invocation halts and nothing newer has started yet), but a halt
    sentinel written into an OLDER turn directory is still a genuine
    unresolved halt even after a newer turn exists: if a resume for turn N
    is still in flight when turn N+1 starts (see clarify.py's
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

    active_dir = active_sentinel_dir(run_uuid)
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

    rj = read_json(run_dir / "run.json")
    if rj:
        return rj.get("status", "running")
    if run_uuid in state.active_runs:
        fut = state.active_runs[run_uuid].get("future")
        if fut and fut.running():
            return "running"
        if fut and fut.done():
            return "complete" if not fut.exception() else "error"
    return "unknown"


def extract_reply_text(final_state: dict) -> str:
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
        data = read_json(Path(final_output_path))
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