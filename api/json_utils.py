"""
api/json_utils.py — small JSON read/write helpers shared across routers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_run_json(
    run_dir: Path, run_uuid: str, mode: Optional[str], status: str,
    error_detail: Optional[str] = None, profile: Optional[str] = None,
) -> None:
    """
    Guards against a stuck background thread waking up (e.g. finishing a
    very long LLM call) after DELETE /run/{run_uuid} has already rmtree'd
    this run — see runs.remove_run's cancelled-run branch. Without this
    check, the write below would silently recreate run_dir with a single
    orphaned run.json in it, making a deleted run reappear.
    """
    if not run_dir.exists():
        log.info(
            "write_run_json: run_dir for %s no longer exists (likely deleted "
            "after cancellation) — discarding status write '%s'", run_uuid, status,
        )
        return
    path = run_dir / "run.json"
    data = read_json(path) or {}
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
    # — previously every "except Exception as exc: write_run_json(...,
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
