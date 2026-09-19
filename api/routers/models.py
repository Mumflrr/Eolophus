"""
api/routers/models.py — model listing, load/unload, and role reassignment.
"""

from __future__ import annotations

import asyncio

import yaml
from fastapi import APIRouter, HTTPException

from api.paths import PROJECT_ROOT
from api.state import executor

router = APIRouter()


@router.get("/models")
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


@router.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Manually trigger a model load (swaps out current model)."""
    def _load():
        from clients.model_manager import ensure_model_loaded
        ensure_model_loaded(model_id)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _load)
    return {"status": "loaded", "model_id": model_id}


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Stop the currently loaded model server."""
    from clients.model_manager import stop_all, current_model
    if current_model() != model_id:
        raise HTTPException(status_code=400, detail=f"{model_id} is not currently loaded")
    stop_all()
    return {"status": "unloaded", "model_id": model_id}


@router.patch("/models/{model_id}/role/{role}")
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
