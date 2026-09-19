"""
api/state.py — process-wide shared state.

Single-run executor and the in-memory _active_runs registry, used by
runs.py, chat.py, and clarify.py to submit background work and to look
up/store a run's env_overrides for a later resume. This must be ONE
shared instance across all routers (not one per module) — the whole
point of _active_runs is that a POST /clarify or POST /chat handler can
see the future a POST /run handler started.

_env_lock is likewise re-exported (not redefined) from clients.llm — see
that module's env_lock docstring for why this must be the same lock
object pipeline/graph.py's truncation-retry wrapper uses, rather than a
fresh threading.Lock() per module that imports it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from clients.llm import env_lock as env_lock  # re-exported, same instance

# Single-run executor — one pipeline at a time on a single GPU
executor = ThreadPoolExecutor(max_workers=1)

# run_uuid -> {"future": Future, "run_dir": str, "env_overrides": dict}
active_runs: dict[str, Any] = {}

# Statuses under which a run is considered "busy" — POST /chat, DELETE
# /chat, and DELETE /run all gate on this same set so a background
# thread never gets a file or directory pulled out from under it.
ACTIVE_STATUSES = {"running", "waiting_for_clarification", "waiting_for_truncation_retry"}
