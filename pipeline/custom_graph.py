"""
pipeline/custom_graph.py — compiles a PipelineDefinition into a runnable
LangGraph StateGraph.

This is the piece that turns "data describing a pipeline" into "an actual
executable graph." Kept deliberately separate from graph.py (the built-in
pipeline) — the built-in pipeline stays hand-wired and untouched; custom
pipelines are compiled fresh from their definition file.

Caching: one compiled graph per pipeline name, built once and reused
until the definition file changes (mtime-checked, not just cached forever
— editing a pipeline in the GUI should take effect on the next run
without a server restart).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.custom_nodes import make_freeform_node, make_decision_node, make_decision_router
from pipeline.custom_validator import validate_pipeline_definition
from schemas.pipeline_def import (
    PipelineDefinition, PipelineStep, ExistingStep, FreeformStep, DecisionStep, StepType
)

log = logging.getLogger(__name__)

PIPELINES_DIR = Path(__file__).parent.parent / "config" / "pipelines"
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)

# ── Existing-node function registry ────────────────────────────────────────
# Deferred import inside a function to avoid import-time circularity with
# nodes/ (which itself imports clients.llm, pipeline.state, etc.)

def _existing_node_registry() -> dict[str, Callable]:
    from nodes.classifier import classify_node
    from nodes.vision import vision_decode_node
    from nodes.ideation import ideation_node
    from nodes.planner import plan_node
    from nodes.drafter import draft_node, draft_short_node
    from nodes.appraiser import appraise_node
    from nodes.bugfixer import bugfix_node
    from nodes.critics import critic_a_node, critic_b_node
    from nodes.validator import synthesise_node, validate_node
    from nodes.describe import describe_node
    from nodes.distiller import distiller_node

    return {
        "classify":     classify_node,
        "vision_decode":vision_decode_node,
        "ideation":     ideation_node,
        "plan":         plan_node,
        "draft":        draft_node,
        "draft_short":  draft_short_node,
        "appraise":     appraise_node,
        "bugfix":       bugfix_node,
        "critic_a":     critic_a_node,
        "critic_b":     critic_b_node,
        "synthesise":   synthesise_node,
        "validate":     validate_node,
        "describe":     describe_node,
        "distiller":    distiller_node,
    }


def _wrap_existing_node(step: ExistingStep, base_fn: Callable) -> Callable:
    """
    Wraps a built-in node function to optionally apply this step's model/
    budget overrides for the duration of the call.

    SAFETY ASSUMPTION: this relies on os.environ being effectively
    single-threaded from the pipeline's perspective — true today because
    api/server.py's executor is ThreadPoolExecutor(max_workers=1) (one
    GPU, one run at a time) and custom pipelines have no fan-out (steps
    are strictly sequential per PipelineDefinition's design). If either
    of those ever changes — multi-GPU concurrent runs, or a future
    parallel-branch primitive added to PipelineDefinition — this env-var
    approach becomes a race condition and must be replaced with a
    contextvars-based or explicit-parameter override threaded through
    call_role's signature instead. Flagging here so future-me doesn't
    "clean this up" without noticing why it was safe before.
    """
    if step.model_override is None and step.budget_override is None:
        return base_fn   # no override — use the real function unmodified

    def _wrapped(state: dict) -> dict:
        import os
        # llm.py's call_role() reads PIPELINE_STEP_MODEL_OVERRIDE and
        # PIPELINE_STEP_BUDGET_OVERRIDE immediately after role resolution
        # (see call_role in clients/llm.py) and applies them for exactly
        # this one call. Scoped here via try/finally so it can never leak
        # into a sibling node's call even if this one raises.
        prev_model  = os.environ.get("PIPELINE_STEP_MODEL_OVERRIDE")
        prev_budget = os.environ.get("PIPELINE_STEP_BUDGET_OVERRIDE")
        try:
            if step.model_override:
                os.environ["PIPELINE_STEP_MODEL_OVERRIDE"] = step.model_override
            if step.budget_override is not None:
                os.environ["PIPELINE_STEP_BUDGET_OVERRIDE"] = str(step.budget_override)
            return base_fn(state)
        finally:
            if prev_model is None:
                os.environ.pop("PIPELINE_STEP_MODEL_OVERRIDE", None)
            else:
                os.environ["PIPELINE_STEP_MODEL_OVERRIDE"] = prev_model
            if prev_budget is None:
                os.environ.pop("PIPELINE_STEP_BUDGET_OVERRIDE", None)
            else:
                os.environ["PIPELINE_STEP_BUDGET_OVERRIDE"] = prev_budget

    return _wrapped


# ── Total-iteration safety net ──────────────────────────────────────────────

def _make_global_cap_wrapper(node_fn: Callable, definition: PipelineDefinition) -> Callable:
    """
    Wraps ANY node with a check against max_total_iterations — the
    combined safety net across all loop-backs in the pipeline, independent
    of individual decision nodes' own caps. Increments a single shared
    counter on every node visit.
    """
    def _wrapped(state: dict) -> dict:
        count = state.get("_global_step_count", 0) + 1
        if count > definition.max_total_iterations:
            log.error(
                "Pipeline '%s' exceeded max_total_iterations (%d) — forcing halt",
                definition.name, definition.max_total_iterations,
            )
            return {
                "pipeline_complete": True,
                "pipeline_failed":   True,
                "failure_reason": (
                    f"max_total_iterations ({definition.max_total_iterations}) exceeded"
                ),
                "_global_step_count": count,
            }
        result = node_fn(state)
        result["_global_step_count"] = count
        return result

    return _wrapped


# ── Graph compilation ──────────────────────────────────────────────────────

_compiled_cache: dict[str, tuple[float, object, object]] = {}   # name -> (mtime, saver_cm, compiled_app)


def load_pipeline_definition(name: str) -> PipelineDefinition:
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No custom pipeline named '{name}' at {path}")

    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineDefinition.model_validate(raw)


def get_custom_graph(name: str):
    """
    Returns a compiled, runnable graph for the named custom pipeline.
    Recompiles automatically if the definition file's mtime changed since
    the last build — so editing a pipeline in the GUI takes effect on the
    NEXT run without restarting the server.
    """
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No custom pipeline named '{name}'")

    mtime = path.stat().st_mtime
    cached = _compiled_cache.get(name)
    if cached and cached[0] == mtime:
        return cached[2]

    # If this pipeline was previously compiled under an older definition,
    # close its old SqliteSaver connection before rebuilding — otherwise
    # editing a pipeline repeatedly leaks open sqlite connections.
    if cached:
        old_saver_cm = cached[1]
        try:
            old_saver_cm.__exit__(None, None, None)
        except Exception as e:
            log.warning("Error closing previous SqliteSaver for '%s': %s", name, e)

    definition = load_pipeline_definition(name)
    errors = validate_pipeline_definition(definition)
    if errors:
        raise ValueError(
            f"Pipeline '{name}' failed validation:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    app, saver_cm = _compile(definition)
    _compiled_cache[name] = (mtime, saver_cm, app)
    log.info("Compiled custom pipeline '%s' (%d steps)", name, len(definition.steps))
    return app


def _compile(definition: PipelineDefinition):
    builder = StateGraph(PipelineState)
    existing_fns = _existing_node_registry()

    step_by_id: dict[str, PipelineStep] = {s.id: s for s in definition.steps}
    step_order = [s.id for s in definition.steps]

    # ── Add every node ────────────────────────────────────────────────────
    for step in definition.steps:
        if step.type == StepType.EXISTING:
            base_fn = existing_fns[step.node_name]
            fn = _wrap_existing_node(step, base_fn)
        elif step.type == StepType.FREEFORM:
            fn = make_freeform_node(step)
        elif step.type == StepType.DECISION:
            fn = make_decision_node(step)
        else:
            raise ValueError(f"Unknown step type: {step.type}")

        fn = _make_global_cap_wrapper(fn, definition)
        builder.add_node(step.id, fn)

    # ── Wire edges ──────────────────────────────────────────────────────────
    for i, step in enumerate(definition.steps):
        if step.type == StepType.DECISION:
            # make_decision_router already resolves outcome -> next_step,
            # INCLUDING the max_iterations cap override — it returns a real
            # step id or LangGraph's END sentinel directly, not the raw
            # decision value. So the edge map passed to add_conditional_edges
            # is an identity map: every possible resolved target maps to
            # itself. This differs from mapping outcome.value -> target,
            # which would prevent the cap override from ever substituting a
            # different target than what the raw decision selected.
            router = make_decision_router(step)

            possible_targets: set = set()
            for outcome in step.outcomes:
                possible_targets.add(END if outcome.next_step == "__end__" else outcome.next_step)
            identity_edge_map = {t: t for t in possible_targets}

            builder.add_conditional_edges(step.id, router, identity_edge_map)
        else:
            # Explicit override takes precedence; otherwise linear "next in list"
            override = definition.edge_overrides.get(step.id)
            if override:
                target = END if override == "__end__" else override
                builder.add_edge(step.id, target)
            elif i + 1 < len(step_order):
                builder.add_edge(step.id, step_order[i + 1])
            else:
                builder.add_edge(step.id, END)

    builder.set_entry_point(definition.entry_step)

    checkpoint_path = Path.home() / ".pipeline" / "custom_checkpoints.db"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # SqliteSaver.from_conn_string returns a context manager, not a saver
    # directly (confirmed against the installed langgraph-checkpoint-sqlite
    # API). We enter it manually and keep it open for the lifetime of this
    # cached compiled graph — get_custom_graph() closes it when a pipeline
    # definition changes and needs recompiling.
    saver_cm = SqliteSaver.from_conn_string(str(checkpoint_path))
    memory   = saver_cm.__enter__()

    app = builder.compile(checkpointer=memory)
    return app, saver_cm
