"""
api/server.py — FastAPI server for the Eolophus pipeline.

Exposes all pipeline functionality over HTTP so the web UI, macOS SwiftUI app,
and the chess Bluetooth bridge can all talk to the same localhost server.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Dependencies (add to requirements.txt):
    fastapi uvicorn[standard] sse-starlette psutil
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

API_DIR      = Path(__file__).parent
PROJECT_ROOT = API_DIR.parent
RUNS_DIR     = Path(os.environ.get("PIPELINE_RUNS_DIR", PROJECT_ROOT / "runs"))
STATIC_DIR   = API_DIR / "static"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Eolophus Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-run executor — one pipeline at a time on a single GPU
_executor   = ThreadPoolExecutor(max_workers=1)
_active_runs: dict[str, Any] = {}   # run_uuid → {future, env_overrides}
_env_lock   = threading.Lock()       # guards process-global env var mutations


# ── Request / response schemas ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    task:          str
    mode:          Optional[str] = None   # "short" | "long" | "ultra" | None (auto)
    task_type:     Optional[str] = None   # "coding" | "ideation" | "mixed" | "describe"
    no_ensemble:   bool = False
    use_search:    bool = False
    pipeline:      Optional[str] = None   # None = built-in pipeline; else a name
                                            # from config/pipelines/{name}.yaml,
                                            # created via the pipeline CRUD endpoints below.
                                            # mode/task_type/no_ensemble are ignored when
                                            # pipeline is set — custom pipelines define
                                            # their own model/budget choices per step.


class PipelineStepIn(BaseModel):
    """Wire-format mirror of schemas.pipeline_def's step union, for the CRUD API."""
    type: str    # "existing" | "freeform" | "decision"
    id:   str

    # existing
    node_name:      Optional[str] = None
    model_override: Optional[str] = None
    budget_override:Optional[int] = None

    # freeform
    model:         Optional[str] = None
    budget_tokens: Optional[int] = None
    thinking:      Optional[bool] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    input_key:     Optional[str] = None
    output_key:    Optional[str] = None
    feedback_mode: Optional[str] = None   # "auto" | "none"

    # decision
    outcomes:       Optional[list[dict]] = None   # [{value, next_step, description?}]
    is_loop_back:   Optional[bool] = None
    max_iterations: Optional[int] = None


class PipelineDefIn(BaseModel):
    """Wire-format for creating/updating a custom pipeline definition."""
    name:                 str
    description:          str = ""
    entry_step:            str
    steps:                 list[PipelineStepIn]
    edge_overrides:         dict[str, str] = {}
    max_total_iterations:   int = 20


class ClarifyRequest(BaseModel):
    answer: str


class BudgetPatch(BaseModel):
    budgets: dict[str, int]   # stage → token budget (-1 = unlimited)


class ChessRequest(BaseModel):
    movePlayed:    Optional[str]   = None
    side:          Optional[str]   = None
    moveNotation:  Optional[str]   = None
    classification:Optional[str]  = None
    cpLoss:        Optional[int]   = None
    bestMove:      Optional[str]   = None
    bestMoveEval:  Optional[float] = None
    evalAfter:     Optional[float] = None
    gamePhase:     Optional[str]   = None
    winPctWhite:   Optional[float] = None
    winPctDraw:    Optional[float] = None
    winPctBlack:   Optional[float] = None
    materialDelta: Optional[int]   = None
    depthProfile:  Optional[str]   = None
    tacticalFlags: list[str]       = []
    bestLine:      list[str]       = []
    pieces:        Optional[dict]  = None
    slow_mode:     bool            = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_run_dir(run_uuid: str) -> Path:
    return RUNS_DIR / run_uuid


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_status(run_uuid: str) -> str:
    """Derive run status from sentinel files in the run directory."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        return "not_found"
    if (run_dir / "clarification.json").exists():
        return "waiting_for_clarification"
    rj = _read_json(run_dir / "run.json")
    if rj:
        return rj.get("status", "running")
    if run_uuid in _active_runs:
        fut = _active_runs[run_uuid].get("future")
        if fut and fut.running():
            return "running"
        if fut and fut.done():
            return "complete" if not fut.exception() else "error"
    return "unknown"


def _get_runs_from_db(limit: int = 100) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _get_lessons_from_db(
    task_type: Optional[str] = None,
    issue_category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 200,
) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        conditions = ["confidence_score >= ?"]
        params: list[Any] = [min_confidence]
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if issue_category:
            conditions.append("issue_category = ?")
            params.append(issue_category)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM lessons WHERE {where} ORDER BY confidence_score DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result
    finally:
        conn.close()


def _load_routing_config() -> dict:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _save_routing_config(cfg: dict) -> None:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── Pipeline runner (runs in ThreadPoolExecutor) ──────────────────────────────

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
                   last run (edits made via the CRUD endpoints below take
                   effect on the NEXT run with no server restart needed).
    """
    run_dir = Path(initial_state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with _env_lock:
        # Save original values and set overrides
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

        config = {"configurable": {"thread_id": run_uuid}}
        if callbacks:
            config["callbacks"] = callbacks

        _write_run_json(run_dir, run_uuid, initial_state.get("mode"), "running")
        final_state = app_graph.invoke(initial_state, config=config)

        status = "unresolvable" if final_state.get("pipeline_failed") else "complete"
        _write_run_json(run_dir, run_uuid, initial_state.get("mode"), status)
        update_run_status(run_uuid, status)
        return final_state

    except Exception as exc:
        _write_run_json(run_dir, run_uuid, initial_state.get("mode"), "error")
        try:
            from storage.critique_store import update_run_status
            update_run_status(run_uuid, "error")
        except Exception:
            pass
        raise

    finally:
        with _env_lock:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _write_run_json(run_dir: Path, run_uuid: str, mode: Optional[str], status: str) -> None:
    path = run_dir / "run.json"
    data = _read_json(path) or {}
    data.update({"run_uuid": run_uuid, "mode": mode, "status": status})
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Run management endpoints ───────────────────────────────────────────────────
@app.get("/debug/run/{run_uuid}/exception")
async def debug_run_exception(run_uuid: str):
    info = _active_runs.get(run_uuid)
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

@app.post("/run")
async def start_run(req: RunRequest):
    """Start a new pipeline run. Returns immediately with run_uuid."""
    run_uuid = str(uuid.uuid4())
    run_dir  = get_run_dir(run_uuid)
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_state: dict = {
        "run_uuid":          run_uuid,
        "run_dir":           str(run_dir),
        "iteration":         0,
        "is_sub_spec":       False,
        "decompose":         False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
        "raw_text_input":    req.task,
        "normalised_input":  req.task,
        "use_search":        req.use_search,
    }

    env_overrides: dict[str, str] = {}

    if req.pipeline:
        # Custom pipeline: mode/task_type/no_ensemble/ultra are concepts
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
        if req.mode and req.mode != "auto":
            initial_state["mode"] = req.mode
        if req.task_type:
            initial_state["task_type"] = req.task_type
        if req.no_ensemble:
            env_overrides["PIPELINE_NO_ENSEMBLE"] = "1"
        if req.mode == "ultra":
            env_overrides["PIPELINE_ULTRA"] = "1"
        elif req.mode == "short":
            env_overrides["PIPELINE_FORCE_SHORT"] = "1"

    # Write initial run.json immediately so the UI can poll it
    _write_run_json(run_dir, run_uuid, req.mode, "running")

    # Register in DB
    try:
        from storage.critique_store import write_run
        write_run(
            run_uuid    = run_uuid,
            mode        = req.pipeline or req.mode or "auto",
            task_type   = req.task_type or "auto",
            complexity  = "auto",
            is_sub_spec = False,
        )
    except Exception as e:
        log.warning("Failed to write run to DB: %s", e)

    future = _executor.submit(
        _run_pipeline_thread, run_uuid, initial_state, env_overrides, req.pipeline
    )
    _active_runs[run_uuid] = {"future": future, "run_dir": str(run_dir)}

    return {
        "run_uuid": run_uuid,
        "status":   "running",
        "run_dir":  str(run_dir),
        "pipeline": req.pipeline or "built-in",
    }


@app.get("/runs")
async def list_runs(limit: int = 50):
    """List recent pipeline runs from the database."""
    db_runs = _get_runs_from_db(limit)
    # Enrich with live status for active runs
    for r in db_runs:
        uid = r.get("run_uuid")
        if uid in _active_runs:
            r["status"] = _run_status(uid)
    return db_runs


@app.get("/run/{run_uuid}")
async def get_run(run_uuid: str):
    """Full run detail: artifacts, stages log, current status."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts: dict = {}
    for fname in [
        "run.json", "classification.json", "planspec.json", "draft.json",
        "appraisal_report.json", "fixed.json", "critique.json",
        "verdict.json", "final.json", "final_validation.json",
    ]:
        data = _read_json(run_dir / fname)
        if data is not None:
            artifacts[fname.replace(".json", "")] = data

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

    clarification = _read_json(run_dir / "clarification.json")

    return {
        "run_uuid":      run_uuid,
        "status":        _run_status(run_uuid),
        "artifacts":     artifacts,
        "stages":        stages,
        "clarification": clarification,
    }


@app.post("/clarify/{run_uuid}")
async def clarify_run(run_uuid: str, req: ClarifyRequest):
    """Resume a pipeline that has halted for clarification."""
    run_dir = get_run_dir(run_uuid)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    # Remove clarification sentinel
    clarification_file = run_dir / "clarification.json"
    if clarification_file.exists():
        clarification_file.unlink()

    env_overrides = _active_runs.get(run_uuid, {}).get("env_overrides", {})

    def _resume():
        from pipeline.graph import get_graph
        from langgraph.types import Command
        app_graph, callbacks = get_graph()
        config = {"configurable": {"thread_id": run_uuid}}
        if callbacks:
            config["callbacks"] = callbacks
        return app_graph.invoke(Command(resume=req.answer), config=config)

    future = _executor.submit(_resume)
    _active_runs[run_uuid] = {"future": future, "run_dir": str(run_dir)}
    return {"status": "resumed"}


_ACTIVE_STATUSES = {"running", "waiting_for_clarification"}


@app.delete("/run/{run_uuid}")
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
    status  = _run_status(run_uuid)

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

    if status in _ACTIVE_STATUSES:
        run_dir.mkdir(parents=True, exist_ok=True)   # in case it's mid-creation
        _write_run_json(run_dir, run_uuid, None, "cancelled")
        try:
            from storage.critique_store import update_run_status
            update_run_status(run_uuid, "cancelled")
        except Exception:
            pass
        _active_runs.pop(run_uuid, None)
        return {"status": "cancelled", "run_uuid": run_uuid}

    # Finished (complete / unresolvable / error / cancelled) -> hard delete
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

    _active_runs.pop(run_uuid, None)
    return {"status": "deleted", "run_uuid": run_uuid}


# ── SSE stream endpoint ────────────────────────────────────────────────────────

@app.get("/stream/{run_uuid}")
async def stream_run(run_uuid: str):
    """
    Server-Sent Events stream of stages.log entries for a run.
    The client receives one JSON object per completed stage.
    Sends {type: 'complete'} or {type: 'clarification', question: '...'} as sentinels.
    """
    async def generator():
        run_dir    = get_run_dir(run_uuid)
        log_path   = run_dir / "stages.log"
        clarify_path = run_dir / "clarification.json"

        # Wait for log file (up to 15s)
        for _ in range(150):
            if log_path.exists() or not run_dir.exists():
                break
            await asyncio.sleep(0.1)

        if not log_path.exists():
            yield f"data: {json.dumps({'type': 'error', 'message': 'log not found'})}\n\n"
            return

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

            # Check for clarification halt
            if clarify_path.exists():
                data = _read_json(clarify_path) or {}
                yield f"data: {json.dumps({'type': 'clarification', **data})}\n\n"
                return

            # Check for completion
            status = _run_status(run_uuid)
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


# ── Model management endpoints ────────────────────────────────────────────────

@app.get("/models")
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


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Manually trigger a model load (swaps out current model)."""
    def _load():
        from clients.model_manager import ensure_model_loaded
        ensure_model_loaded(model_id)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load)
    return {"status": "loaded", "model_id": model_id}


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Stop the currently loaded model server."""
    from clients.model_manager import stop_all, current_model
    if current_model() != model_id:
        raise HTTPException(status_code=400, detail=f"{model_id} is not currently loaded")
    stop_all()
    return {"status": "unloaded", "model_id": model_id}


@app.patch("/models/{model_id}/role/{role}")
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


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.get("/config/budgets")
async def get_budgets():
    """Return current thinking_budgets from routing.yaml."""
    cfg = _load_routing_config()
    return cfg.get("thinking_budgets", {})


@app.patch("/config/budgets")
async def update_budgets(req: BudgetPatch):
    """Update thinking_budgets in routing.yaml."""
    cfg = _load_routing_config()
    budgets = cfg.setdefault("thinking_budgets", {})
    for stage, tokens in req.budgets.items():
        if tokens < -1:
            raise HTTPException(status_code=400, detail=f"Invalid budget {tokens} for {stage}")
        budgets[stage] = tokens
    _save_routing_config(cfg)
    return {"status": "updated", "budgets": budgets}


@app.post("/config/reload")
async def reload_config():
    """Bust all in-memory config caches. Call after manual YAML edits."""
    from clients.llm import _config_cache, _prompt_cache
    _config_cache.clear()
    _prompt_cache.clear()
    return {"status": "reloaded"}


@app.get("/config/searxng")
async def get_searxng_url():
    """Return current SearXNG URL."""
    return {"url": os.environ.get("SEARXNG_URL", "http://localhost:8888")}


@app.post("/config/searxng")
async def set_searxng_url(body: dict):
    """Update SearXNG URL for this server session."""
    url = body.get("url", "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    os.environ["SEARXNG_URL"] = url
    return {"status": "updated", "url": url}


# ── Lesson endpoints ──────────────────────────────────────────────────────────

@app.get("/lessons")
async def get_lessons(
    task_type:      Optional[str]   = None,
    issue_category: Optional[str]   = None,
    min_confidence: float           = 0.0,
    limit:          int             = 100,
):
    """List lessons with optional filters."""
    return _get_lessons_from_db(task_type, issue_category, min_confidence, limit)


@app.delete("/lessons/{lesson_uuid}")
async def delete_lesson(lesson_uuid: str):
    """Remove a lesson from the store."""
    from storage.db import get_conn
    conn = get_conn()
    try:
        conn.execute("DELETE FROM lessons WHERE lesson_uuid = ?", (lesson_uuid,))
        conn.commit()
        if conn.execute(
            "SELECT changes() as n"
        ).fetchone()["n"] == 0:
            raise HTTPException(status_code=404, detail="Lesson not found")
    finally:
        conn.close()
    return {"status": "deleted", "lesson_uuid": lesson_uuid}


@app.post("/lessons/distill")
async def trigger_distillation():
    """Manually trigger meta-distillation of accumulated lessons."""
    # Placeholder — meta-distillation node not yet implemented
    return {"status": "not_implemented", "message": "Meta-distillation coming soon"}


# ── Chess endpoint ─────────────────────────────────────────────────────────────

@app.post("/chess/analyse")
async def chess_analyse(req: ChessRequest):
    """
    Analyse a chess move. Called by the Bluetooth bridge and Swift apps.
    Returns headline, explanation, suggestion, tacticalPattern.
    fast mode: 512 thinking tokens
    slow mode: 1024 thinking tokens (pass slow_mode=true for blunders/sacrifices)
    """
    run_uuid = str(uuid.uuid4())
    run_dir  = RUNS_DIR / "chess" / run_uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    request_dict = req.model_dump(exclude={"slow_mode"})

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use default executor (not the pipeline executor)
        lambda: analyse_chess_move(request_dict, str(run_dir), req.slow_mode),
    )
    return result


# ── Custom pipeline CRUD ────────────────────────────────────────────────────
# Create/list/edit/delete/validate custom pipeline definitions
# (config/pipelines/{name}.yaml). Every write goes through the same
# validate_pipeline_definition() the compiler itself uses (pipeline/
# custom_validator.py), so a definition that fails to save would also
# have failed to compile — the GUI never gets a false "saved OK" that
# then blows up at run time.

def _wire_to_definition(body: PipelineDefIn) -> "PipelineDefinition":
    """Convert the API's flat wire format into the real tagged-union schema."""
    from schemas.pipeline_def import (
        PipelineDefinition, ExistingStep, FreeformStep, DecisionStep,
        DecisionOutcome, StepType,
    )

    steps = []
    for s in body.steps:
        if s.type == "existing":
            if not s.node_name:
                raise HTTPException(400, f"Step '{s.id}': type=existing requires node_name")
            steps.append(ExistingStep(
                id=s.id, node_name=s.node_name,
                model_override=s.model_override, budget_override=s.budget_override,
            ))
        elif s.type == "freeform":
            if not s.model or not s.system_prompt or not s.output_key:
                raise HTTPException(
                    400, f"Step '{s.id}': type=freeform requires model, system_prompt, output_key"
                )
            steps.append(FreeformStep(
                id=s.id, model=s.model,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                thinking=s.thinking or False,
                system_prompt=s.system_prompt,
                user_template=s.user_template or "{input}",
                input_key=s.input_key or "normalised_input",
                output_key=s.output_key,
                feedback_mode=s.feedback_mode or "auto",
            ))
        elif s.type == "decision":
            if not s.system_prompt or not s.outcomes:
                raise HTTPException(
                    400, f"Step '{s.id}': type=decision requires system_prompt, outcomes"
                )
            steps.append(DecisionStep(
                id=s.id, model=s.model or "9b",
                thinking=s.thinking or False,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                system_prompt=s.system_prompt,
                input_key=s.input_key or "normalised_input",
                outcomes=[DecisionOutcome(**o) for o in s.outcomes],
                is_loop_back=s.is_loop_back or False,
                max_iterations=s.max_iterations,
                feedback_mode=s.feedback_mode or "auto",
            ))
        else:
            raise HTTPException(400, f"Step '{s.id}': unknown type '{s.type}'")

    return PipelineDefinition(
        name=body.name, description=body.description,
        entry_step=body.entry_step, steps=steps,
        edge_overrides=body.edge_overrides,
        max_total_iterations=body.max_total_iterations,
    )


@app.get("/pipelines")
async def list_pipelines():
    """List all saved custom pipeline definitions (name + description only)."""
    from pipeline.custom_graph import PIPELINES_DIR, load_pipeline_definition

    result = []
    for path in sorted(PIPELINES_DIR.glob("*.yaml")):
        try:
            defn = load_pipeline_definition(path.stem)
            result.append({
                "name": defn.name, "description": defn.description,
                "step_count": len(defn.steps), "entry_step": defn.entry_step,
            })
        except Exception as e:
            result.append({"name": path.stem, "description": f"[INVALID: {e}]", "step_count": 0})
    return result


@app.get("/pipelines/{name}")
async def get_pipeline(name: str):
    """Full definition for one custom pipeline — the shape a GUI editor needs."""
    from pipeline.custom_graph import load_pipeline_definition
    try:
        defn = load_pipeline_definition(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    return json.loads(defn.model_dump_json())


@app.post("/pipelines/validate")
async def validate_pipeline(body: PipelineDefIn):
    """
    Validate a pipeline definition WITHOUT saving it. Lets a GUI show
    live errors while the user is still editing, before they commit.
    """
    from pipeline.custom_validator import validate_pipeline_definition
    try:
        defn = _wire_to_definition(body)   # raises HTTPException on structural issues
    except HTTPException:
        raise
    except Exception as e:
        # Pydantic validation errors (duplicate ids, missing max_iterations
        # on a self-declared loop-back, bad entry_step, etc.)
        return {"valid": False, "errors": [str(e)]}

    errors = validate_pipeline_definition(defn)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/pipelines")
async def create_pipeline(body: PipelineDefIn):
    """
    Create or overwrite a custom pipeline definition. Validates before
    writing — a pipeline that fails validation is never saved to disk,
    so config/pipelines/ never contains a definition that would fail to
    compile at run time.
    """
    from pipeline.custom_graph import PIPELINES_DIR
    from pipeline.custom_validator import validate_pipeline_definition

    defn = _wire_to_definition(body)
    errors = validate_pipeline_definition(defn)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Pipeline failed validation, not saved", "errors": errors},
        )

    path = PIPELINES_DIR / f"{body.name}.yaml"
    import yaml
    with open(path, "w") as f:
        yaml.dump(json.loads(defn.model_dump_json()), f, default_flow_style=False, sort_keys=False)

    return {"status": "saved", "name": body.name, "path": str(path)}


@app.delete("/pipelines/{name}")
async def delete_pipeline(name: str):
    """Delete a custom pipeline definition. Does not affect past runs."""
    from pipeline.custom_graph import PIPELINES_DIR
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    path.unlink()
    return {"status": "deleted", "name": name}


@app.get("/pipelines/nodes/available")
async def list_available_node_types():
    """
    What a GUI needs to populate an 'add step' dropdown: every reusable
    existing-node name, and the step-type shapes (freeform/decision)
    with their required fields, sourced directly from the real schema
    so this never drifts out of sync with what the backend actually accepts.
    """
    from schemas.pipeline_def import EXISTING_NODE_NAMES

    return {
        "existing_nodes": EXISTING_NODE_NAMES,
        "step_types": {
            "existing": {
                "required": ["id", "node_name"],
                "optional": ["model_override", "budget_override"],
            },
            "freeform": {
                "required": ["id", "model", "system_prompt", "output_key"],
                "optional": ["budget_tokens", "thinking", "user_template",
                             "input_key", "feedback_mode"],
            },
            "decision": {
                "required": ["id", "system_prompt", "outcomes"],
                "optional": ["model", "thinking", "budget_tokens", "input_key",
                             "is_loop_back", "max_iterations", "feedback_mode"],
                "outcome_shape": {"value": "string", "next_step": "string (step id or __end__)",
                                   "description": "string, optional"},
            },
        },
    }


# ── Health endpoint ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Server health, active model, and queue status."""
    from clients.model_manager import current_model, _is_port_alive
    from clients.llm import _load_config

    cfg    = _load_config()
    hot    = current_model()
    hot_cfg = cfg["models"].get(hot, {}) if hot else {}

    active_count = sum(
        1 for r in _active_runs.values()
        if (f := r.get("future")) and f.running()
    )

    return {
        "status":       "ok",
        "hot_model":    hot,
        "hot_port":     hot_cfg.get("port"),
        "hot_context":  hot_cfg.get("context_len"),
        "active_runs":  active_count,
        "queue_depth":  _executor._work_queue.qsize(),
        "searxng_url":  os.environ.get("SEARXNG_URL", "http://localhost:8888"),
    }


# ── Static / UI ───────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not built. Place index.html in api/static/"}
    return FileResponse(str(index))


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    from storage.db import initialise
    initialise()
    log.info("Eolophus API server started. UI: http://localhost:8000")
