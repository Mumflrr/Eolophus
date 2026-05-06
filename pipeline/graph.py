"""
pipeline/graph.py — LangGraph graph definition.

Wires all nodes and routers into a compiled StateGraph.
Langfuse callback is attached at compile time.
The graph is compiled once at module load and reused across runs.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from langgraph.checkpoint.memory import MemorySaver
from nodes.gatekeeper import gatekeeper_node
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from nodes import (
    classify_node,
    vision_decode_node,
    ideation_node,
    plan_node,
    draft_node,
    draft_short_node,
    appraise_node,
    bugfix_node,
    critic_a_node,
    critic_b_node,
    synthesise_node,
    validate_node,
    final_validate_node,
)
from pipeline.routers import (
    route_after_input,
    route_after_vision,
    route_after_classify,
    route_after_ideation,
    route_after_plan,
    route_after_draft_guard,
    route_after_draft_short_guard,
    route_after_bugfix,
    route_after_critic_a,
    route_after_critic_b,
    route_after_synthesise,
    route_after_validate,
    route_after_sub_specs,
    route_after_final_validate,
)
from nodes.describe import describe_node



log = logging.getLogger(__name__)


def clarify_node(state):
    return {
        "pipeline_failed":  True,
        "failure_reason":   state.get("clarification_question", "Clarification needed."),
        "pipeline_complete": True,
    }

# ── Sub-spec runner node ──────────────────────────────────────────────────────

def sub_spec_runner_node(state: PipelineState) -> dict:
    """
    Orchestrates sub-spec decomposition.
    Spawns individual pipeline runs for each sub-spec,
    collects SubSpecInterfaces, and prepares for final_validate.

    Each sub-spec run is a fresh pipeline invocation (short mode)
    with the component's spec as its task input.
    """
    import uuid as _uuid
    from pathlib import Path
    from schemas.sub_spec import SubSpecInterface, InterfaceStatus

    run_dir    = state["run_dir"]
    plan       = state.get("plan_spec")
    run_uuid   = state["run_uuid"]

    if not plan:
        raise ValueError("sub_spec_runner_node: no plan_spec in state")

    sub_specs_dir = Path(run_dir) / "sub_specs"
    sub_specs_dir.mkdir(exist_ok=True)

    sub_spec_uuids     = []
    sub_spec_interfaces = []

    for component in plan.components:
        sub_uuid = str(_uuid.uuid4())
        sub_dir  = sub_specs_dir / sub_uuid
        sub_dir.mkdir()

        # Write interface placeholder
        iface = SubSpecInterface(
            sub_spec_uuid    = sub_uuid,
            parent_run_uuid  = run_uuid,
            component_name   = component.name,
            status           = InterfaceStatus.PENDING,
            inputs           = component.interface_inputs,
            outputs          = component.interface_outputs,
            implementation_path = str(sub_dir / "fixed.json"),
        )

        # Build sub-spec task input from ComponentSpec
        task_input = (
            f"Implement the following component as part of a larger system.\n\n"
            f"Component: {component.name}\n"
            f"Responsibility: {component.responsibility}\n"
            f"Inputs: {', '.join(component.interface_inputs) or 'none'}\n"
            f"Outputs: {', '.join(component.interface_outputs) or 'none'}\n"
            f"Dependencies: {', '.join(component.dependencies) or 'none'}\n\n"
            f"Full spec:\n{component.model_dump_json(indent=2)}\n\n"
            f"MoE routing context: {plan.moe_routing_context}\n"
            f"Edge cases to handle: {'; '.join(plan.edge_cases)}"
        )

        # Run sub-spec pipeline
        log.info("Running sub-spec %s for component '%s'", sub_uuid[:8], component.name)
        sub_result = _run_sub_spec(
            sub_uuid   = sub_uuid,
            sub_dir    = str(sub_dir),
            task_input = task_input,
            parent_uuid= run_uuid,
        )

        # Update interface status
        iface = iface.model_copy(update={
            "status": InterfaceStatus.COMPLETE if sub_result else InterfaceStatus.FAILED
        })

        sub_spec_uuids.append(sub_uuid)
        sub_spec_interfaces.append(iface)
        log.info(
            "Sub-spec %s ('%s'): %s",
            sub_uuid[:8], component.name,
            iface.status
        )

    return {
        "sub_spec_uuids":      sub_spec_uuids,
        "sub_spec_interfaces": sub_spec_interfaces,
    }


def _run_sub_spec(
    sub_uuid:   str,
    sub_dir:    str,
    task_input: str,
    parent_uuid:str,
) -> bool:
    """
    Run a complete short-mode pipeline for a single sub-spec component.
    Returns True if the sub-spec completed successfully.
    """
    import json
    from pathlib import Path

    # Initial state for sub-spec run
    initial_state: PipelineState = {
        "run_uuid":        sub_uuid,
        "run_dir":         sub_dir,
        "mode":            "short",
        "task_type":       "coding",
        "is_sub_spec":     True,
        "parent_run_uuid": parent_uuid,
        "iteration":       0,
        "raw_text_input":  task_input,
        "normalised_input":task_input,
        "decompose":       False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
    }

    try:
        app = get_graph()
        final_state = app.invoke(initial_state)
        return not final_state.get("pipeline_failed", True)
    except Exception as e:
        log.error("Sub-spec %s failed: %s", sub_uuid[:8], e)
        return False


# ── Graph construction ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    """
    Build and compile the LangGraph StateGraph.
    Cached — compiled once per process.
    """
    builder = StateGraph(PipelineState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("vision_decode",   vision_decode_node)
    builder.add_node("gatekeeper", gatekeeper_node)
    builder.add_node("classify",        classify_node)
    builder.add_node("ideation",        ideation_node)
    builder.add_node("plan",            plan_node)
    builder.add_node("draft",           draft_node)
    builder.add_node("draft_short",     draft_short_node)
    builder.add_node("appraise",        appraise_node)
    builder.add_node("bugfix",          bugfix_node)
    builder.add_node("critic_a",        critic_a_node)
    builder.add_node("critic_b",        critic_b_node)
    builder.add_node("synthesise",      synthesise_node)
    builder.add_node("validate",        validate_node)
    builder.add_node("sub_spec_runner", sub_spec_runner_node)
    builder.add_node("final_validate",  final_validate_node)
    builder.add_node("describe",        describe_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_conditional_entry_point(
        route_after_input,
        {
            "vision_decode": "vision_decode",
            "classify":      "classify",
        },
    )

    # ── Vision → classify ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "vision_decode",
        route_after_vision,
        {"classify": "classify"},
    )

    # ── Classify → ideation or plan ───────────────────────────────────────────
    builder.add_conditional_edges("classify", route_after_classify, {
        "describe":  "describe",
        "clarify":   "clarify",
        "ideation":  "ideation",
        "plan":      "plan",
    })

    # ── Ideation → plan ───────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "ideation",
        route_after_ideation,
        {"plan": "plan"},
    )

    # ── Plan → draft / draft_short / sub_spec_runner ─────────────────────────
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "draft":           "draft",
            "draft_short":     "draft_short",
            "sub_spec_runner": "sub_spec_runner",
        },
    )

    # ── Draft (long mode) → guard → appraise or redraft ──────────────────────
    builder.add_conditional_edges(
        "draft",
        route_after_draft_guard,
        {
            "appraise": "appraise",
            "draft":    "draft",     # lazy eval loop
        },
    )

    # ── Draft (short mode) → guard → bugfix or redraft ───────────────────────
    builder.add_conditional_edges(
        "draft_short",
        route_after_draft_short_guard,
        {
            "bugfix":     "bugfix",
            "draft_short":"draft_short",
        },
    )

    # ── Appraise → bugfix (always) ────────────────────────────────────────────
    builder.add_edge("appraise", "bugfix")

    # ── Bugfix → critic_a or gatekeeper ────────────────────────────────────────
    builder.add_conditional_edges(
        "bugfix",
        route_after_bugfix,
        {
            "critic_a": "critic_a",
            "validate": "gatekeeper",
        },
    )

    # ── Critic A → critic_b or synthesise ────────────────────────────────────
    builder.add_conditional_edges(
        "critic_a",
        route_after_critic_a,
        {
            "critic_b":  "critic_b",
            "synthesise":"synthesise",
        },
    )

    # ── Critic B → synthesise ────────────────────────────────────────────────
    builder.add_conditional_edges(
        "critic_b",
        route_after_critic_b,
        {"synthesise": "synthesise"},
    )

    # ── Synthesise → gatekeeper ──────────────────────────────────────────────
    builder.add_conditional_edges(
        "synthesise",
        route_after_synthesise,
        {"validate": "gatekeeper"},
    )

    # ── Validate → end / plan / draft / bugfix ───────────────────────────────
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "distiller": "distiller",
            "plan": "plan",
            "draft": "draft",
            "bugfix": "bugfix"
        },
    )

    # ── Sub-spec runner → final_validate ─────────────────────────────────────
    builder.add_conditional_edges(
        "sub_spec_runner",
        route_after_sub_specs,
        {"final_validate": "final_validate"},
    )

    # ── Final validate → distiller ───────────────────────────────────────────
    # We change this routing logic. Instead of going to END, it goes to distiller.
    builder.add_conditional_edges(
        "final_validate",
        route_after_final_validate,  # Ensure this function returns "distiller" on success instead of END
        {"distiller": "distiller", "end": END},
    )

    # ── Add the new Distiller Node ───────────────────────────────────────────
    from nodes.distiller import distiller_node
    builder.add_node("distiller", distiller_node)
    builder.add_edge("distiller", END)

    builder.add_edge("describe", END)
    builder.add_node("clarify", clarify_node)
    builder.add_edge("clarify", END)
    
    # ── Gatekeeper always flows to Validate ──────────────────────────────────
    builder.add_edge("gatekeeper", "validate")

    # ── Compile with Langfuse callback and MemorySaver ───────────────────────
    callbacks = _build_callbacks()
    
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory)

    log.info("Pipeline graph compiled (%d nodes)", len(builder.nodes))
    return app


def _build_callbacks() -> list:
    """Build Langfuse callback handler if configured."""
    callbacks = []
    try:
        from langfuse.langchain import CallbackHandler
        import os
        
        # 1. Fallback to localhost if not in your .env, and forcefully set it in os.environ
        langfuse_host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        os.environ["LANGFUSE_HOST"] = langfuse_host
        
        # 2. Initialize with NO arguments (Langfuse v3/v4 requirement)
        handler = CallbackHandler()
        
        callbacks.append(handler)
        log.info("Langfuse callback attached at %s", langfuse_host)
    except ImportError as e:
        log.warning(
            f"langfuse import failed ({e}) — tracing disabled. "
            "Make sure you are using 'langfuse.langchain' for v3/v4."
        )
    except Exception as e:
        log.warning("Failed to attach Langfuse callback: %s", e)
    return callbacks