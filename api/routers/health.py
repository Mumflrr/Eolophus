"""
api/routers/health.py — server health, active model, and queue status.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from api.state import active_runs, executor

router = APIRouter()


@router.get("/health")
async def health():
    """Server health, active model, and queue status."""
    from clients.model_manager import current_model
    from clients.llm import _load_config

    cfg    = _load_config()
    hot    = current_model()
    hot_cfg = cfg["models"].get(hot, {}) if hot else {}

    active_count = sum(
        1 for r in active_runs.values()
        if (f := r.get("future")) and f.running()
    )

    return {
        "status":       "ok",
        "hot_model":    hot,
        "hot_port":     hot_cfg.get("port"),
        "hot_context":  hot_cfg.get("context_len"),
        "active_runs":  active_count,
        "queue_depth":  executor._work_queue.qsize(),
        "searxng_url":  os.environ.get("SEARXNG_URL", "http://localhost:8888"),
    }
