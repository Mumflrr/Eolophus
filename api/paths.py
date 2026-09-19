"""
api/paths.py — filesystem layout for pipeline runs.

Every router that touches a run's on-disk artifacts imports from here
rather than recomputing RUNS_DIR or reimplementing the turn-directory
resolution logic. Keeping this in one place is what makes the
clarify/truncation "which directory is this halt actually in" fix (see
_active_sentinel_dir) apply consistently everywhere it's checked:
runs.py, chat.py, clarify.py, and the SSE stream all call the same
function instead of five slightly-different reimplementations drifting
apart over time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

API_DIR      = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent
RUNS_DIR     = Path(os.environ.get("PIPELINE_RUNS_DIR", PROJECT_ROOT / "runs"))
STATIC_DIR   = API_DIR / "static"

RUNS_DIR.mkdir(parents=True, exist_ok=True)


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
    deliberately NOT part of this — see chat.py's _run_chat_turn_thread,
    which always writes run.json to the top-level run_dir, never the
    turn dir.
    """
    return get_run_dir(run_uuid) / "turns" / str(seq)


def latest_turn_seq(run_uuid: str) -> Optional[int]:
    """Highest numbered subdirectory under run_dir/turns/, or None if there
    isn't one yet (i.e. this run has had no chat follow-ups)."""
    turns_dir = get_run_dir(run_uuid) / "turns"
    if not turns_dir.exists():
        return None
    seqs = [int(p.name) for p in turns_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(seqs) if seqs else None


def active_sentinel_dir(run_uuid: str) -> Path:
    """
    Directory a currently-running graph invocation is actually writing
    its stage artifacts and halt sentinels (clarification.json,
    truncated.json) into — the top-level run_dir for the original run,
    or run_dir/turns/<latest_seq>/ once at least one chat follow-up has
    started (see get_chat_turn_dir / chat.py's _run_chat_replan's
    turn_run_dir).

    This exists because clarify_node and the truncation-retry wrapper
    (pipeline/graph.py) both write their sentinel via Path(state["run_dir"]),
    and state["run_dir"] IS the turn directory for a chat follow-up, not
    the top-level run_dir — status.py's _run_status/_read_artifacts_from
    previously only ever checked the top-level directory, which happened
    to be invisible for clarification halts on turn 2+ (same latent bug
    this fixes for the new truncated.json sentinel) since nothing
    exercised that path loudly enough to notice: a clarification halt on
    a later turn would simply never report "waiting_for_clarification",
    it would fall through to whatever run.json's stale status said
    instead.
    """
    seq = latest_turn_seq(run_uuid)
    if seq is not None:
        return get_chat_turn_dir(run_uuid, seq)
    return get_run_dir(run_uuid)
