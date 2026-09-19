"""
api/routers/config.py — routing.yaml budgets, config-cache reload, and
the SearXNG URL setting.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from api.routing_config import load_routing_config, save_routing_config
from api.schemas import BudgetPatch

router = APIRouter()


@router.get("/config/budgets")
async def get_budgets():
    """Return current thinking_budgets from routing.yaml."""
    cfg = load_routing_config()
    return cfg.get("thinking_budgets", {})


@router.patch("/config/budgets")
async def update_budgets(req: BudgetPatch):
    """Update thinking_budgets in routing.yaml."""
    cfg = load_routing_config()
    budgets = cfg.setdefault("thinking_budgets", {})
    for stage, tokens in req.budgets.items():
        if tokens < -1:
            raise HTTPException(status_code=400, detail=f"Invalid budget {tokens} for {stage}")
        budgets[stage] = tokens
    save_routing_config(cfg)
    return {"status": "updated", "budgets": budgets}


@router.post("/config/reload")
async def reload_config():
    """Bust all in-memory config caches. Call after manual YAML edits."""
    from clients.llm import _config_cache, _prompt_cache
    _config_cache.clear()
    _prompt_cache.clear()
    return {"status": "reloaded"}


@router.get("/config/searxng")
async def get_searxng_url():
    """Return current SearXNG URL."""
    return {"url": os.environ.get("SEARXNG_URL", "http://localhost:8888")}


@router.post("/config/searxng")
async def set_searxng_url(body: dict):
    """Update SearXNG URL for this server session."""
    url = body.get("url", "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    os.environ["SEARXNG_URL"] = url
    return {"status": "updated", "url": url}
