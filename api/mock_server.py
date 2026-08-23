"""
api/mock_server.py — standalone fixture server for GUI development.

The real server.py requires a running pipeline: local model servers,
a populated SQLite db, actual VRAM. None of that exists in a fresh chat
building a frontend. This file implements the EXACT SAME route contract
(see api_contract.md) but returns realistic fixture data and simulates
run progression in-memory — no external dependencies beyond FastAPI.

Run:
    pip install fastapi uvicorn[standard]
    uvicorn api.mock_server:app --host 0.0.0.0 --port 8000 --reload

Then point the GUI at http://localhost:8000 exactly as it would point at
the real server — the contract is identical. Swap mock_server for server
when real backend access is available; no frontend code should need to
change.

── Test triggers (type these into the task submission field) ──────────────
  Task text containing "clarify"  → run halts asking a clarification question
  Task text containing "fail"     → run ends with status "unresolvable"
  Task text containing "slow"     → run takes ~25s instead of ~8s (test
                                     long-running UI states / queue_depth)
  mode="ultra"                    → run takes ~15s and simulates a model
                                     swap in the stage log
  Anything else                   → normal ~8s run through all stages,
                                     ends "complete"
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

API_DIR    = Path(__file__).parent
STATIC_DIR = API_DIR / "static"

app = FastAPI(title="Eolophus Pipeline API (MOCK)", version="1.0.0-mock")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ═══════════════════════════════════════════════════════════════════════════
# In-memory state
# ═══════════════════════════════════════════════════════════════════════════

_runs: dict[str, dict] = {}          # run_uuid → run state dict
_run_locks: dict[str, asyncio.Event] = {}   # run_uuid → "clarified" event

_HOT_MODEL = {"id": "9b", "since": time.time()}

_STAGE_SEQUENCE = [
    ("classify", "Qwen3.5-9B",      600,  120, 0.0),
    ("plan",     "Qwen3.5-9B",      1400, 380, 0.35),
    ("draft",    "Qwen3.6-35B-MoE", 3200, 1100, 0.55),
    ("appraise", "DeepCoder-14B",   1800, 420, 0.40),
    ("bugfix",   "DeepCoder-14B",   900,  310, 0.0),
    ("critic_a", "Qwen3.5-9B",      500,  90,  0.0),
    ("critic_b", "DeepCoder-14B",   1200, 260, 0.30),
    ("synthesise","Qwen3.5-9B",     400,  140, 0.0),
    ("validate", "Qwen3.5-9B",      450,  100, 0.0),
    ("distill",  "Qwen3.5-9B",      300,  60,  0.0),
]

_LESSON_TAGS_POOL = [
    ["python", "async", "error_handling"], ["fastapi", "pydantic", "rest"],
    ["typescript", "react"], ["database", "sqlalchemy"], ["testing", "pytest"],
    ["docker", "cli"], ["python", "class"], ["async", "sqlalchemy"],
]
_ISSUE_CATEGORIES = [
    "logic_error", "constraint_violation", "spec_delta",
    "missing_requirement", "error_handling", "type_error", "edge_case",
]


def _seed_lessons() -> list[dict]:
    lessons = []
    templates = [
        ("Never use SQLAlchemy 2.0 Session.execute() without explicit "
         "transaction context — wrap in 'with session.begin()' to avoid "
         "implicit autocommit causing data loss on exception.",
         "Missing transaction context caused silent partial writes."),
        ("Always validate FastAPI path params with a Pydantic constrained "
         "type rather than a raw str — prevents downstream 500s from "
         "malformed UUIDs reaching the database layer.",
         "Raw string path param passed directly to a UUID lookup crashed."),
        ("Never mutate a Pydantic model's list field in place — use "
         "model_copy(update=...) or the mutation won't be reflected in "
         "downstream state after LangGraph merges the node's return dict.",
         "In-place list append was silently dropped by state merge."),
        ("Always set explicit timeout on httpx.AsyncClient calls in "
         "async route handlers — the default has no timeout and can hang "
         "a worker indefinitely on a stalled upstream.",
         "Unbounded httpx call blocked the event loop for 90+ seconds."),
        ("Never assume asyncio.gather() preserves exception order — "
         "wrap each coroutine with a per-task error handler if you need "
         "to know which specific call failed.",
         "gather() masked which of 5 parallel calls actually raised."),
    ]
    for i, (resolution, summary) in enumerate(templates):
        lessons.append({
            "lesson_uuid":        str(uuid.uuid4()),
            "source_run_uuid":    str(uuid.uuid4()),
            "issue_summary":      summary,
            "resolution_pattern": resolution,
            "example_context":    None,
            "task_type":          "coding",
            "tags":               random.choice(_LESSON_TAGS_POOL),
            "model_caught":       random.choice(["deepcoder-14b", "qwen3.5-9b"]),
            "issue_category":     random.choice(_ISSUE_CATEGORIES),
            "confidence_score":   round(random.uniform(1.0, 4.5), 1),
            "times_seen":         random.randint(1, 6),
            "times_retrieved":    random.randint(0, 12),
            "times_useful":       random.randint(0, 8),
            "last_triggered":     "2026-08-10T14:22:00",
            "is_meta_lesson":     0,
            "source_lesson_uuids":[],
            "created_at":         "2026-07-28T09:00:00",
            "updated_at":         "2026-08-10T14:22:00",
        })
    return lessons

_LESSONS = _seed_lessons()


def _model_fixture() -> list[dict]:
    hot = _HOT_MODEL["id"]
    base = [
        {"id": "9b", "name": "Qwen3.5-9B", "quant": "Q6_K", "source": "bartowski",
         "port": 8081, "base_url": "http://localhost:8081/v1", "context_len": 32768,
         "roles": ["classify", "plan", "critic_a", "validate", "vision_decode",
                    "synthesis_simple", "draft_short", "describe", "distill",
                    "gatekeeper", "chess_fast", "chess_slow"],
         "thinking_default": False, "mtp": False},
        {"id": "35b", "name": "Qwen3.6-35B-MoE", "quant": "UD-Q4_K_M", "source": "unsloth",
         "port": 8083, "base_url": "http://localhost:8083/v1", "context_len": 65536,
         "roles": ["draft", "synthesis_complex"], "thinking_default": True, "mtp": False},
        {"id": "27b", "name": "Qwen3.5-27B", "quant": "IQ2_XXS", "source": "bartowski",
         "port": 8082, "base_url": "http://localhost:8082/v1", "context_len": 16384,
         "roles": ["ideation"], "thinking_default": False, "mtp": False},
        {"id": "deepcoder", "name": "DeepCoder-14B", "quant": "Q4_K_M", "source": "bartowski",
         "port": 8084, "base_url": "http://localhost:8084/v1", "context_len": 16384,
         "roles": ["appraise", "bugfix", "critic_b"], "thinking_default": True, "mtp": False},
        {"id": "27b_ultra", "name": "Qwen3.6-27B-MTP-Ultra", "quant": "Q4_K_M-MTP",
         "source": "havenoammo/froggeric", "port": 8085,
         "base_url": "http://localhost:8085/v1", "context_len": 131072,
         "roles": ["ultra_plan", "ultra_draft", "ultra_appraise", "ultra_critic"],
         "thinking_default": True, "mtp": True},
    ]
    for m in base:
        m["status"] = "loaded" if m["id"] == hot else "unloaded"
        m["is_hot"] = m["id"] == hot
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Schemas (mirrors server.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    task:        str
    mode:        Optional[str] = None
    task_type:   Optional[str] = None
    no_ensemble: bool = False
    use_search:  bool = False
    pipeline:    Optional[str] = None   # None = built-in pipeline; else a saved custom pipeline name

class ClarifyRequest(BaseModel):
    answer: str


# ── Pipeline wire-format models ──────────────────────────────────────────────
# Mirrors server.py's PipelineStepIn / PipelineDefIn exactly (same field
# names, same optionality) so a GUI built against this mock needs zero
# changes when pointed at the real backend.

class PipelineStepIn(BaseModel):
    type: str    # "existing" | "freeform" | "decision"
    id:   str

    # existing
    node_name:       Optional[str] = None
    model_override:  Optional[str] = None
    budget_override: Optional[int] = None

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
    name:                  str
    description:           str = ""
    entry_step:            str
    steps:                 list[PipelineStepIn]
    edge_overrides:        dict[str, str] = {}
    max_total_iterations:  int = 20

class BudgetPatch(BaseModel):
    budgets: dict[str, int]

class ChessRequest(BaseModel):
    movePlayed:    Optional[str]   = None
    side:          Optional[str]   = None
    moveNotation:  Optional[str]   = None
    classification:Optional[str]   = None
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


_DEFAULT_BUDGETS = {
    "plan": 1024, "draft": 4096, "draft_pass2": 0, "appraise": 2048,
    "bugfix": 0, "critic_a": 0, "critic_b": 2048, "synthesise": 0,
    "validate": 0, "final_validate": 0, "classify": 0, "audit": 0,
    "describe": 512, "ideation": 0, "vision": 0, "distill": 0,
    "chess_fast": 512, "chess_slow": 1024,
    "ultra_plan": -1, "ultra_draft": -1, "ultra_appraise": -1, "ultra_critic": -1,
}
_budgets_state = dict(_DEFAULT_BUDGETS)


# ═══════════════════════════════════════════════════════════════════════════
# Run simulation
# ═══════════════════════════════════════════════════════════════════════════

async def _simulate_run(run_uuid: str, task: str, mode: Optional[str]):
    run = _runs[run_uuid]
    slow    = "slow" in task.lower()
    should_fail = "fail" in task.lower()
    should_clarify = "clarify" in task.lower()
    is_ultra = mode == "ultra"

    speed = 3.0 if slow else (1.6 if is_ultra else 1.0)

    if is_ultra:
        run["stages"].append({
            "ts": _now(), "stage": "model_swap", "model": "Qwen3.6-27B-MTP-Ultra",
            "prompt_hash": "swap", "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 4200, "load_ms": 4200, "ttft_ms": 0, "think_ratio": 0,
            "status": "ok", "retries": 0,
        })
        await asyncio.sleep(1.2 * speed)

    for i, (stage, model, tin, tout, think) in enumerate(_STAGE_SEQUENCE):
        if run["status"] == "cancelled":
            return

        # Clarification trigger fires after "plan" stage
        if should_clarify and stage == "plan":
            run["stages"].append(_stage_entry(stage, model, tin, tout, think))
            run["clarification"] = {
                "question": "Should the rate limiter persist state across "
                             "process restarts, or is in-memory only sufficient?"
            }
            run["status"] = "waiting_for_clarification"
            event = _run_locks.setdefault(run_uuid, asyncio.Event())
            await event.wait()
            event.clear()
            run["clarification"] = None
            continue

        await asyncio.sleep(random.uniform(0.4, 0.9) * speed)
        run["stages"].append(_stage_entry(stage, model, tin, tout, think))

        # skip ensemble stages for short-mode-flavored runs (simple heuristic)
        if not is_ultra and i == 4 and random.random() < 0.3:
            break

    if should_fail:
        run["status"] = "unresolvable"
        run["artifacts"]["verdict"] = {
            "category": "unresolvable", "synthesis_model": "9b",
            "description": "Iteration limit reached without a passing verdict.",
            "specific_issues": ["Simulated failure — task text contained 'fail'"],
        }
    else:
        run["status"] = "complete"
        run["artifacts"]["final"] = {
            "answer": f"[MOCK] Completed pipeline run for: {task[:80]}",
            "task_type": run.get("task_type") or "coding",
        }
        run["artifacts"]["verdict"] = {
            "category": "pass", "synthesis_model": "9b",
            "description": "Output satisfies the plan.", "specific_issues": [],
        }


def _stage_entry(stage, model, tin, tout, think) -> dict:
    jitter = lambda v: int(v * random.uniform(0.85, 1.15))
    return {
        "ts": _now(), "stage": stage, "model": model,
        "prompt_hash": uuid.uuid4().hex[:12],
        "tokens_in": jitter(tin), "tokens_out": jitter(tout),
        "latency_ms": round(jitter(tin + tout) * 1.8, 1),
        "load_ms": 0.0, "ttft_ms": round(random.uniform(80, 400), 1),
        "think_ratio": think, "status": "ok", "retries": 0,
    }


async def _simulate_custom_pipeline_run(run_uuid: str, task: str, pipeline_name: str):
    """
    Walks a saved PipelineDefIn-shaped definition step by step, producing a
    realistic stage log — decision steps actually pick a random outcome and
    the walk follows it (respecting max_iterations / max_total_iterations),
    freeform/existing steps just log a stage entry and move on. This is the
    interim mock approach the handoff doc suggested: reuse the same
    stage-progression simulator, relabeled from the real definition rather
    than the fixed built-in _STAGE_SEQUENCE.
    """
    run = _runs[run_uuid]
    should_fail = "fail" in task.lower()
    should_clarify = "clarify" in task.lower()
    defn = _PIPELINES[pipeline_name]

    step_by_id = {s["id"]: s for s in defn["steps"]}
    step_order = [s["id"] for s in defn["steps"]]
    index_of = {sid: i for i, sid in enumerate(step_order)}
    edge_overrides = defn.get("edge_overrides", {})
    max_total = defn.get("max_total_iterations", 20)
    iter_counts: dict[str, int] = {}

    current = defn["entry_step"]
    visited_total = 0
    clarified_once = False

    while current and current != "__end__":
        if run["status"] == "cancelled":
            return
        visited_total += 1
        if visited_total > max_total:
            run["status"] = "unresolvable"
            run["artifacts"]["verdict"] = {
                "category": "unresolvable", "synthesis_model": "9b",
                "description": f"max_total_iterations ({max_total}) exceeded.",
                "specific_issues": [f"Pipeline '{pipeline_name}' did not terminate within its cap."],
            }
            return

        step = step_by_id.get(current)
        if step is None:
            break

        model = step.get("model") or step.get("model_override") or "9b"
        tokens_in = random.randint(200, 900)
        tokens_out = random.randint(150, 700)
        think = 0.3 if step.get("thinking") else 0.0

        # Clarification trigger — fires once, on the first freeform/existing
        # step, same convention as the built-in pipeline's mock behavior.
        if should_clarify and not clarified_once and step["type"] != "decision":
            run["stages"].append(_stage_entry(step["id"], model, tokens_in, tokens_out, think))
            clarified_once = True
            run["clarification"] = {
                "question": f"Step '{step['id']}' needs more direction — "
                             f"can you clarify what '{task[:60]}' should emphasize?"
            }
            run["status"] = "waiting_for_clarification"
            event = _run_locks.setdefault(run_uuid, asyncio.Event())
            await event.wait()
            event.clear()
            run["clarification"] = None

        await asyncio.sleep(random.uniform(0.35, 0.75))

        if step["type"] == "decision":
            outcomes = step.get("outcomes") or []
            chosen = random.choice(outcomes) if outcomes else None
            entry = _stage_entry(step["id"], model, tokens_in // 2, 60, think)
            entry["stage"] = f"decision:{step['id']}"
            if chosen:
                entry["tokens_out"] = 60
            run["stages"].append(entry)

            if not chosen:
                current = None
                continue

            if step.get("is_loop_back"):
                iter_counts[step["id"]] = iter_counts.get(step["id"], 0) + 1
                cap = step.get("max_iterations")
                if cap is not None and iter_counts[step["id"]] >= cap:
                    # Force the pipeline's own cap behavior: pick any
                    # outcome whose target differs from the chosen one,
                    # same "break the loop" logic as custom_nodes.py.
                    alt = next((o for o in outcomes if o.get("next_step") != chosen.get("next_step")), None)
                    chosen = alt or chosen
            current = chosen.get("next_step")
        else:
            run["stages"].append(_stage_entry(step["id"], model, tokens_in, tokens_out, think))
            override = edge_overrides.get(step["id"])
            if override:
                current = override
            else:
                idx = index_of[step["id"]]
                current = step_order[idx + 1] if idx + 1 < len(step_order) else "__end__"

    if should_fail:
        run["status"] = "unresolvable"
        run["artifacts"]["verdict"] = {
            "category": "unresolvable", "synthesis_model": "9b",
            "description": "Iteration limit reached without a passing verdict.",
            "specific_issues": ["Simulated failure — task text contained 'fail'"],
        }
    else:
        run["status"] = "complete"
        run["artifacts"]["final"] = {
            "answer": f"[MOCK] Completed custom pipeline '{pipeline_name}' for: {task[:80]}",
            "pipeline": pipeline_name,
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════════
# Custom pipelines — in-memory store standing in for config/pipelines/*.yaml
# ═══════════════════════════════════════════════════════════════════════════
# Ported from pipeline/custom_validator.py's five checks, adapted to work
# against the plain-dict wire format (PipelineDefIn.model_dump()) instead of
# the real Pydantic model tree, since this mock can't import pipeline/ or
# schemas/ (real backend packages) without dragging in the whole pipeline.
# Same checks, same error text shape, so GUI validation feedback matches
# what the real server would say.

_PIPELINES: dict[str, dict] = {}   # name -> PipelineDefIn.model_dump()-shaped dict

EXISTING_NODE_NAMES = [
    "classify", "vision_decode", "ideation", "plan",
    "draft", "draft_short", "appraise", "bugfix",
    "critic_a", "critic_b", "synthesise", "validate",
    "describe", "distiller",
]

_EXISTING_OUTPUT_KEYS = {
    "classify":      {"classification", "mode", "task_type"},
    "vision_decode": {"visual_description", "normalised_input"},
    "ideation":      {"ideation_output"},
    "plan":          {"plan_spec"},
    "draft":         {"draft_output"},
    "draft_short":   {"draft_output"},
    "appraise":      {"appraisal_report"},
    "bugfix":        {"fixed_output"},
    "critic_a":      {"critique_record"},
    "critic_b":      {"critique_record"},
    "synthesise":    {"validation_verdict"},
    "validate":      {"validation_verdict"},
    "describe":      {"final_output_path"},
    "distiller":     set(),
}
_INITIAL_STATE_KEYS = {"raw_text_input", "normalised_input", "raw_image_path"}


def _pydantic_shape_errors(body: PipelineDefIn) -> list[str]:
    """
    Field-level checks that server.py gets for free from Pydantic's
    validators on the real ExistingStep/FreeformStep/DecisionStep/
    PipelineDefinition classes (schemas/pipeline_def.py). Reimplemented
    here by hand since the mock works from the flat PipelineStepIn shape,
    not the real tagged-union models.
    """
    errors: list[str] = []
    ids = [s.id for s in body.steps]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append(f"Duplicate step ids: {dupes}")

    step_ids = set(ids)
    if body.entry_step not in step_ids:
        errors.append(f"entry_step '{body.entry_step}' is not a defined step")

    for s in body.steps:
        if s.type == "existing":
            if not s.node_name:
                errors.append(f"Step '{s.id}': type=existing requires node_name")
            elif s.node_name not in EXISTING_NODE_NAMES:
                errors.append(f"Step '{s.id}': '{s.node_name}' is not a reusable node. Valid: {EXISTING_NODE_NAMES}")
        elif s.type == "freeform":
            if not s.model or not s.system_prompt or not s.output_key:
                errors.append(f"Step '{s.id}': type=freeform requires model, system_prompt, output_key")
        elif s.type == "decision":
            if not s.system_prompt or not s.outcomes:
                errors.append(f"Step '{s.id}': type=decision requires system_prompt, outcomes")
            else:
                if len(s.outcomes) < 2:
                    errors.append(f"Decision step '{s.id}' needs at least 2 outcomes")
                if s.is_loop_back and not s.max_iterations:
                    errors.append(
                        f"Decision step '{s.id}' is marked is_loop_back=True but has "
                        f"no max_iterations. Every loop-back must declare a cap."
                    )
        else:
            errors.append(f"Step '{s.id}': unknown type '{s.type}'")

    return errors


def _graph_shape_errors(body: PipelineDefIn) -> list[str]:
    """Port of custom_validator.py's five graph-shape checks."""
    errors: list[str] = []
    step_ids = {s.id for s in body.steps}
    step_by_id = {s.id: s for s in body.steps}
    step_order = [s.id for s in body.steps]
    index_of = {sid: i for i, sid in enumerate(step_order)}

    # _check_edge_targets_exist
    for step in body.steps:
        if step.type == "decision":
            for outcome in (step.outcomes or []):
                nxt = outcome.get("next_step")
                if nxt != "__end__" and nxt not in step_ids:
                    errors.append(
                        f"Decision step '{step.id}' outcome '{outcome.get('value')}' routes to "
                        f"'{nxt}', which is not a defined step id"
                    )
    for from_id, to_id in body.edge_overrides.items():
        if from_id not in step_ids:
            errors.append(f"edge_override references unknown source step '{from_id}'")
        if to_id != "__end__" and to_id not in step_ids:
            errors.append(f"edge_override for '{from_id}' targets unknown step '{to_id}'")

    # _check_undeclared_cycles
    for step in body.steps:
        if step.type == "decision":
            for outcome in (step.outcomes or []):
                nxt = outcome.get("next_step")
                if nxt == "__end__":
                    continue
                target_idx = index_of.get(nxt)
                self_idx = index_of.get(step.id)
                is_backward = target_idx is not None and self_idx is not None and target_idx <= self_idx
                if is_backward and not step.is_loop_back:
                    errors.append(
                        f"Decision step '{step.id}' outcome '{outcome.get('value')}' routes "
                        f"backward to '{nxt}' (creates a cycle) but is_loop_back is not "
                        f"set to True. Every cycle must be explicitly marked so "
                        f"max_iterations is enforced."
                    )
    for from_id, to_id in body.edge_overrides.items():
        if to_id == "__end__":
            continue
        from_idx, to_idx = index_of.get(from_id), index_of.get(to_id)
        if from_idx is not None and to_idx is not None and to_idx <= from_idx:
            errors.append(
                f"edge_override '{from_id}' -> '{to_id}' points backward, creating an "
                f"UNCAPPED cycle. Only decision steps (with is_loop_back=True and "
                f"max_iterations) may create cycles. Route through a decision step instead."
            )

    # _check_reachability
    if body.entry_step in step_by_id or True:  # always attempt; entry-missing is reported separately
        visited: set[str] = set()
        stack = [body.entry_step]
        while stack:
            current = stack.pop()
            if current in visited or current not in step_by_id:
                continue
            visited.add(current)
            step = step_by_id[current]
            if step.type == "decision":
                for outcome in (step.outcomes or []):
                    nxt = outcome.get("next_step")
                    if nxt != "__end__":
                        stack.append(nxt)
            else:
                override = body.edge_overrides.get(current)
                if override and override != "__end__":
                    stack.append(override)
                elif not override:
                    idx = index_of[current]
                    if idx + 1 < len(step_order):
                        stack.append(step_order[idx + 1])
        unreached = sorted(set(step_by_id.keys()) - visited)
        for step_id in unreached:
            errors.append(
                f"Step '{step_id}' is unreachable from entry_step "
                f"'{body.entry_step}' — check for a typo in an edge target"
            )

    # _check_freeform_input_keys
    available: set[str] = set(_INITIAL_STATE_KEYS)
    for step in body.steps:
        if step.type in ("freeform", "decision"):
            ik = step.input_key or "normalised_input"
            if ik not in available:
                errors.append(
                    f"Step '{step.id}' reads input_key '{ik}', which no prior step "
                    f"produces (available at this point: {sorted(available)}). Likely "
                    f"a typo, or the step is positioned before its data exists."
                )
        if step.type == "existing" and step.node_name:
            available |= _EXISTING_OUTPUT_KEYS.get(step.node_name, set())
        elif step.type == "freeform" and step.output_key:
            available.add(step.output_key)

    # _check_decision_outcomes_exhaustive
    for step in body.steps:
        if step.type == "decision":
            values = [o.get("value") for o in (step.outcomes or [])]
            if len(values) != len(set(values)):
                errors.append(f"Decision step '{step.id}' has duplicate outcome values: {values}")

    return errors


def _validate_pipeline_def(body: PipelineDefIn) -> list[str]:
    errors = _pydantic_shape_errors(body)
    # Graph-shape checks assume ids/entry_step are already sane — same
    # ordering the real backend uses (field-level errors first).
    if not errors:
        errors = _graph_shape_errors(body)
    return errors


# ═══════════════════════════════════════════════════════════════════════════
# Run endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/run")
async def start_run(req: RunRequest):
    if req.pipeline:
        # Custom pipeline: mode/task_type/no_ensemble are ignored — same
        # fail-fast-before-execution behavior as the real server, so the
        # GUI can trust that a successful response means the name was real.
        if req.pipeline not in _PIPELINES:
            raise HTTPException(
                status_code=404,
                detail=f"No custom pipeline named '{req.pipeline}'. "
                       f"List available pipelines at GET /pipelines.",
            )
        run_uuid = str(uuid.uuid4())
        _runs[run_uuid] = {
            "run_uuid": run_uuid, "status": "running", "mode": "custom",
            "task_type": "custom", "task": req.task, "pipeline": req.pipeline,
            "stages": [], "artifacts": {}, "clarification": None,
            "started_at": _now(), "completed_at": None,
        }
        asyncio.create_task(_simulate_custom_pipeline_run(run_uuid, req.task, req.pipeline))
        return {"run_uuid": run_uuid, "status": "running", "run_dir": f"/mock/runs/{run_uuid}"}

    run_uuid = str(uuid.uuid4())
    _runs[run_uuid] = {
        "run_uuid": run_uuid, "status": "running", "mode": req.mode or "auto",
        "task_type": req.task_type or "coding", "task": req.task,
        "stages": [], "artifacts": {}, "clarification": None,
        "started_at": _now(), "completed_at": None,
    }
    asyncio.create_task(_simulate_run(run_uuid, req.task, req.mode))
    return {"run_uuid": run_uuid, "status": "running", "run_dir": f"/mock/runs/{run_uuid}"}


@app.get("/runs")
async def list_runs(limit: int = 50):
    out = []
    for r in sorted(_runs.values(), key=lambda x: x["started_at"], reverse=True)[:limit]:
        out.append({
            "run_uuid": r["run_uuid"], "mode": r["mode"], "task_type": r["task_type"],
            "complexity": "moderate", "is_sub_spec": 0, "parent_run_uuid": None,
            "status": r["status"], "stage_reached": r["stages"][-1]["stage"] if r["stages"] else None,
            "correction_iterations": 0,
            "total_tokens": sum(s["tokens_in"] + s["tokens_out"] for s in r["stages"]),
            "total_latency_ms": sum(s["latency_ms"] for s in r["stages"]),
            "started_at": r["started_at"], "completed_at": r["completed_at"],
        })
    return out


@app.get("/run/{run_uuid}")
async def get_run(run_uuid: str):
    r = _runs.get(run_uuid)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_uuid": run_uuid, "status": r["status"], "artifacts": r["artifacts"],
        "stages": r["stages"], "clarification": r["clarification"],
    }


@app.post("/clarify/{run_uuid}")
async def clarify_run(run_uuid: str, req: ClarifyRequest):
    r = _runs.get(run_uuid)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    r["status"] = "running"
    r["clarification"] = None
    event = _run_locks.setdefault(run_uuid, asyncio.Event())
    event.set()
    return {"status": "resumed"}


@app.delete("/run/{run_uuid}")
async def cancel_run(run_uuid: str):
    r = _runs.get(run_uuid)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    r["status"] = "cancelled"
    return {"status": "cancelled"}


@app.get("/stream/{run_uuid}")
async def stream_run(run_uuid: str):
    async def generator():
        yielded = 0
        while True:
            r = _runs.get(run_uuid)
            if not r:
                yield f"data: {json.dumps({'type': 'error', 'message': 'not found'})}\n\n"
                return
            for s in r["stages"][yielded:]:
                yield f"data: {json.dumps(s)}\n\n"
                yielded += 1
            if r["status"] == "waiting_for_clarification" and r["clarification"]:
                yield f"data: {json.dumps({'type': 'clarification', **r['clarification']})}\n\n"
                return
            if r["status"] in ("complete", "unresolvable", "error", "cancelled"):
                yield f"data: {json.dumps({'type': 'complete', 'status': r['status']})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model / config / lesson / chess / health endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/models")
async def get_models():
    return _model_fixture()


@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    await asyncio.sleep(1.5)   # simulate load latency
    _HOT_MODEL["id"] = model_id
    return {"status": "loaded", "model_id": model_id}


@app.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    if _HOT_MODEL["id"] != model_id:
        raise HTTPException(status_code=400, detail=f"{model_id} is not currently loaded")
    _HOT_MODEL["id"] = None
    return {"status": "unloaded", "model_id": model_id}


_ROLE_ASSIGNMENTS: dict[str, str] = {}

@app.patch("/models/{model_id}/role/{role}")
async def reassign_role(model_id: str, role: str):
    _ROLE_ASSIGNMENTS[role] = model_id
    return {"status": "updated", "role": role, "model_id": model_id}


@app.get("/config/budgets")
async def get_budgets():
    return _budgets_state


@app.patch("/config/budgets")
async def update_budgets(req: BudgetPatch):
    for k, v in req.budgets.items():
        if v < -1:
            raise HTTPException(status_code=400, detail=f"Invalid budget {v} for {k}")
        _budgets_state[k] = v
    return {"status": "updated", "budgets": _budgets_state}


@app.post("/config/reload")
async def reload_config():
    return {"status": "reloaded"}


# ═══════════════════════════════════════════════════════════════════════════
# Custom pipeline CRUD
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/pipelines")
async def list_pipelines():
    result = []
    for name, defn in sorted(_PIPELINES.items()):
        result.append({
            "name": defn["name"], "description": defn.get("description", ""),
            "step_count": len(defn["steps"]), "entry_step": defn["entry_step"],
        })
    return result


@app.get("/pipelines/{name}")
async def get_pipeline(name: str):
    defn = _PIPELINES.get(name)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    return defn


@app.post("/pipelines/validate")
async def validate_pipeline(body: PipelineDefIn):
    errors = _validate_pipeline_def(body)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/pipelines")
async def create_pipeline(body: PipelineDefIn):
    errors = _validate_pipeline_def(body)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Pipeline failed validation, not saved", "errors": errors},
        )
    _PIPELINES[body.name] = json.loads(body.model_dump_json())
    return {"status": "saved", "name": body.name, "path": f"/mock/config/pipelines/{body.name}.yaml"}


@app.delete("/pipelines/{name}")
async def delete_pipeline(name: str):
    if name not in _PIPELINES:
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    del _PIPELINES[name]
    return {"status": "deleted", "name": name}


@app.get("/pipelines/nodes/available")
async def list_available_node_types():
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


_searxng_url = {"url": "http://localhost:8888"}

@app.get("/config/searxng")
async def get_searxng_url():
    return _searxng_url


@app.post("/config/searxng")
async def set_searxng_url(body: dict):
    url = body.get("url", "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")
    _searxng_url["url"] = url
    return {"status": "updated", "url": url}


@app.get("/lessons")
async def get_lessons(
    task_type: Optional[str] = None, issue_category: Optional[str] = None,
    min_confidence: float = 0.0, limit: int = 100,
):
    out = _LESSONS
    if task_type:
        out = [l for l in out if l["task_type"] == task_type]
    if issue_category:
        out = [l for l in out if l["issue_category"] == issue_category]
    out = [l for l in out if l["confidence_score"] >= min_confidence]
    return sorted(out, key=lambda l: l["confidence_score"], reverse=True)[:limit]


@app.delete("/lessons/{lesson_uuid}")
async def delete_lesson(lesson_uuid: str):
    global _LESSONS
    before = len(_LESSONS)
    _LESSONS = [l for l in _LESSONS if l["lesson_uuid"] != lesson_uuid]
    if len(_LESSONS) == before:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return {"status": "deleted", "lesson_uuid": lesson_uuid}


@app.post("/lessons/distill")
async def trigger_distillation():
    return {"status": "not_implemented", "message": "Meta-distillation coming soon"}


@app.post("/chess/analyse")
async def chess_analyse(req: ChessRequest):
    await asyncio.sleep(0.8 if not req.slow_mode else 1.6)
    patterns = ["fork", "pin", "back_rank", "development", "best_move", "blunder"]
    result = {
        "headline":        f"[MOCK] {req.moveNotation or 'Move'} — {req.classification or 'analysed'}.",
        "explanation":      "The knight repositions to control the center and pressure the "
                             "weak pawn on d5, consistent with the engine's top line.",
        "tacticalPattern":  random.choice(patterns),
    }
    if req.classification not in ("Excellent", "Good"):
        result["suggestion"] = "Consider Nf3 instead, developing while eyeing e5."
    return result


@app.get("/health")
async def health():
    hot = _HOT_MODEL["id"]
    hot_cfg = next((m for m in _model_fixture() if m["id"] == hot), None)
    running = sum(1 for r in _runs.values() if r["status"] == "running")
    return {
        "status": "ok", "hot_model": hot,
        "hot_port": hot_cfg["port"] if hot_cfg else None,
        "hot_context": hot_cfg["context_len"] if hot_cfg else None,
        "active_runs": running, "queue_depth": max(0, running - 1),
        "searxng_url": _searxng_url["url"],
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not built yet. Place index.html in api/static/"}
    return FileResponse(str(index))
