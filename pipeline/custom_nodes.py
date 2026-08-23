"""
pipeline/custom_nodes.py — generic executors for freeform and decision steps.

These are the two functions that make "custom pipelines" possible without
writing new Python per pipeline. Each is instantiated once per step
definition (via functools.partial in custom_graph.py) and becomes a
LangGraph node like any other.

Existing-node reuse does NOT go through here — it calls the real node
functions (plan_node, draft_node, ...) directly, with only their model/
budget optionally overridden. See custom_graph.py for that wiring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, create_model
from langgraph.graph import END

from clients.llm import call_model, get_model_config
from schemas.pipeline_def import FreeformStep, DecisionStep, FeedbackMode

log = logging.getLogger(__name__)


# ── Generic output schemas ─────────────────────────────────────────────────

class FreeformOutput(BaseModel):
    """
    Universal output shape for freeform nodes. Kept intentionally minimal
    so ANY prompt can target it — the content itself is unstructured text
    in `result`; confidence/clarification still work uniformly so the
    halt-on-ambiguity mechanism applies even to nodes nobody wrote Python for.
    """
    result: str = Field(description="The node's output content, in plain text or "
                                     "the format requested by the system prompt")
    confidence: str = Field(
        default="high",
        description="high/medium/low. low halts the pipeline for human input."
    )
    clarification_question: Optional[str] = Field(default=None)


def _build_decision_schema(outcomes: list[str]) -> type[BaseModel]:
    """
    Dynamically build a schema whose `decision` field is constrained to
    exactly the outcome values this step declares. Using Pydantic's
    Literal via create_model rather than a hand-written class per step —
    there is no fixed DecisionOutput class because the valid values differ
    per step.
    """
    from typing import Literal
    literal_type = Literal[tuple(outcomes)]  # type: ignore
    return create_model(
        "DecisionOutput",
        decision=(literal_type, Field(description=f"Must be exactly one of: {outcomes}")),
        reasoning=(str, Field(description="Brief justification for this decision — "
                                            "this text is what flows forward as feedback")),
    )


# ── Helpers shared by both executors ────────────────────────────────────────

def _serialize_state_value(value) -> str:
    """
    Turn a state value into text for a prompt. Existing-node outputs are
    Pydantic objects (PlanSpec, DraftOutput, ...) — serialize as JSON.
    Freeform outputs and raw strings pass through as-is.
    """
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value)


def _resolve_input(state: dict, input_key: str) -> str:
    """
    Resolve a step's input_key against state. Checks real top-level state
    fields first (plan_spec, draft_output, normalised_input, etc. — the
    built-in nodes' fixed output keys), then falls back to
    custom_step_outputs (where freeform steps' output_key values live,
    since those can't be arbitrary top-level TypedDict keys — see the
    note on PipelineState in pipeline/state.py).
    """
    if input_key in state:
        return _serialize_state_value(state[input_key])
    custom_outputs = state.get("custom_step_outputs") or {}
    if input_key in custom_outputs:
        return _serialize_state_value(custom_outputs[input_key])
    return ""


def _resolve_feedback(state: dict, feedback_mode: FeedbackMode) -> str:
    """
    Read the automatic feedback passthrough set by the most recent decision
    node, unless this step opted out. Returns "" when there's nothing to
    show — callers substitute that into {feedback} in their template.
    """
    if feedback_mode == FeedbackMode.NONE:
        return ""
    return state.get("custom_feedback") or ""


# ── Freeform executor ────────────────────────────────────────────────────────

def make_freeform_node(step: FreeformStep):
    """
    Returns a LangGraph-compatible node function closed over this step's
    config. Called once per freeform step at graph-build time.
    """
    def _node(state: dict) -> dict:
        run_dir = state["run_dir"]

        raw_input = _resolve_input(state, step.input_key)
        feedback  = _resolve_feedback(state, step.feedback_mode)

        user_msg = step.user_template.format_map(
            _SafeDict({"input": raw_input, "feedback": feedback})
        )

        messages = [
            {"role": "system", "content": step.system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        result: FreeformOutput = call_model(
            model_id        = step.model,
            messages        = messages,
            response_schema = FreeformOutput,
            stage           = f"custom:{step.id}",
            run_dir         = run_dir,
            thinking        = step.thinking,
            budget_tokens   = step.budget_tokens,
            max_retries     = 0,
        )

        if result.confidence == "low" and result.clarification_question:
            log.warning("Custom step '%s' halted — needs input: %s",
                        step.id, result.clarification_question)
            return {
                "pipeline_halted":      True,
                "clarification_needed": result.clarification_question,
            }

        out_path = Path(run_dir) / f"custom_{step.id}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        log.info("Custom freeform step '%s' complete (%d chars)", step.id, len(result.result))

        # User-defined output_key names cannot be arbitrary top-level state
        # keys — LangGraph's TypedDict-backed state only tracks keys it
        # declared at graph-build time (see pipeline/state.py comment).
        # Freeform output therefore lives INSIDE the fixed
        # custom_step_outputs dict, keyed by output_key, never as a
        # dynamic top-level field. _serialize_state_value's caller
        # (_resolve_input, below) checks this dict for any input_key that
        # isn't a real built-in state field.
        step_outputs = dict(state.get("custom_step_outputs") or {})
        step_outputs[step.output_key] = result.result
        step_outputs[f"{step.output_key}__full"] = result.model_dump()

        return {"custom_step_outputs": step_outputs}

    _node.__name__ = f"freeform_{step.id}"
    return _node


# ── Decision executor ─────────────────────────────────────────────────────────

def make_decision_node(step: DecisionStep):
    """
    Returns a LangGraph-compatible node function. The node itself only
    calls the model and stashes the decision + reasoning in state — actual
    routing happens in the router function built alongside it in
    custom_graph.py (LangGraph nodes and routers are separate callables).
    """
    outcome_values = [o.value for o in step.outcomes]
    schema = _build_decision_schema(outcome_values)

    def _node(state: dict) -> dict:
        run_dir = state["run_dir"]

        raw_input = _resolve_input(state, step.input_key)
        feedback  = _resolve_feedback(state, step.feedback_mode)

        user_content = raw_input
        if feedback:
            user_content = f"[Prior feedback: {feedback}]\n\n{raw_input}"

        messages = [
            {"role": "system", "content": step.system_prompt},
            {"role": "user",   "content": user_content},
        ]

        result = call_model(
            model_id        = step.model,
            messages        = messages,
            response_schema = schema,
            stage           = f"decision:{step.id}",
            run_dir         = run_dir,
            thinking        = step.thinking,
            budget_tokens   = step.budget_tokens,
            max_retries     = 0,
        )

        log.info("Decision '%s' -> %s (%s)", step.id, result.decision, result.reasoning[:80])

        # Iteration bookkeeping for loop-back caps — keyed per decision step
        # id INSIDE custom_iter_counts (a fixed TypedDict field), so
        # multiple loop-backs in one pipeline don't share a counter, and
        # the counts actually survive LangGraph's state merge. A dynamic
        # top-level key like f"_iter_count__{step.id}" would be silently
        # dropped — LangGraph only tracks keys declared on PipelineState
        # at graph-build time (confirmed empirically; see pipeline/state.py).
        iter_counts = dict(state.get("custom_iter_counts") or {})
        iter_counts[step.id] = iter_counts.get(step.id, 0) + 1

        decisions = dict(state.get("custom_decisions") or {})
        decisions[step.id] = result.decision

        update = {
            "custom_decisions":   decisions,
            "custom_iter_counts": iter_counts,
        }

        if step.feedback_mode == FeedbackMode.AUTO:
            update["custom_feedback"] = result.reasoning
        elif step.feedback_mode == FeedbackMode.NONE:
            update["custom_feedback"] = None

        return update

    _node.__name__ = f"decision_{step.id}"
    return _node


def make_decision_router(step: DecisionStep, max_iterations_override: Optional[int] = None):
    """
    Returns the router function LangGraph calls after a decision node to
    pick the next edge. Separate from make_decision_node because LangGraph
    treats node execution and edge routing as distinct callables.

    Returns real step ids or LangGraph's END sentinel — NOT the raw
    "__end__" string from config. The config string is a human-authoring
    convenience; this is the only place it gets translated to what
    LangGraph's conditional-edges API actually expects.

    Reads decision/iteration state from the fixed custom_decisions and
    custom_iter_counts dict fields (NOT dynamic top-level keys — those are
    silently dropped by LangGraph's TypedDict-backed state; see the note
    on PipelineState in pipeline/state.py).
    """
    def _resolve(next_step: str):
        return END if next_step == "__end__" else next_step

    outcome_map = {o.value: _resolve(o.next_step) for o in step.outcomes}
    cap = step.max_iterations if step.is_loop_back else max_iterations_override

    def _router(state: dict) -> str:
        decisions = state.get("custom_decisions") or {}
        decision  = decisions.get(step.id)
        current_target = outcome_map.get(decision)

        if step.is_loop_back and cap is not None:
            iter_counts = state.get("custom_iter_counts") or {}
            count = iter_counts.get(step.id, 0)
            if count >= cap:
                log.warning(
                    "Decision '%s' hit max_iterations (%d) — forcing exit", step.id, cap
                )
                # Route to any outcome whose resolved target differs from
                # the one the raw decision just selected — i.e. break out
                # of whichever specific outcome is causing the loop, rather
                # than assuming the loop-back outcome points at this step's
                # own id (it usually points at an EARLIER step instead).
                for outcome in step.outcomes:
                    resolved = _resolve(outcome.next_step)
                    if resolved != current_target:
                        return resolved
                return END

        if current_target is None:
            log.error(
                "Decision '%s' produced unmapped value '%s' — this should be "
                "impossible given schema constraints. Ending pipeline.",
                step.id, decision,
            )
            return END

        return current_target

    return _router


class _SafeDict(dict):
    """format_map helper — leaves unresolved {keys} intact rather than raising."""
    def __missing__(self, key):
        return "{" + key + "}"
