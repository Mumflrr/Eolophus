"""
pipeline/graph.py — LangGraph graph definition.

Wires all nodes and routers into a compiled StateGraph.
Langfuse callback is attached at compile time.
The graph is compiled once at module load and reused across runs.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Optional
from langgraph.checkpoint.sqlite import SqliteSaver
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
    """
    Halt the pipeline and ask the caller a clarifying question, then
    resume in place once an answer arrives.

    Uses LangGraph's interrupt() so that app_graph.invoke() in
    _run_pipeline_thread returns early (with an "__interrupt__" entry in
    the returned state) instead of running to completion. The graph
    checkpoint (MemorySaver, keyed on thread_id=run_uuid) preserves
    exactly where we are, so POST /clarify/{run_uuid} can later call
    app_graph.invoke(Command(resume=answer), config={"configurable":
    {"thread_id": run_uuid}}) and continue right here with `answer`
    bound to whatever interrupt() returns below.

    We also write clarification.json ourselves — LangGraph's interrupt
    mechanism doesn't know about that convention, but server.py's
    _run_status() and GET /run both read it to report/surface
    "waiting_for_clarification", and /clarify deletes it on resume.

    IMPORTANT: this node must NOT set pipeline_failed / pipeline_complete.
    Those fields tell _run_pipeline_thread the graph is done (successfully
    or not); a clarification halt is neither — it's a pause. Setting them
    here is what previously caused every clarification-needed run to be
    reported as "unresolvable".

    Two different upstream nodes can route here, using two different
    field names for the same concept — classify_node/route_after_classify
    use "clarification_question" (from TaskClassification), while
    plan_node uses "clarification_needed" (from PlanSpec, set alongside
    pipeline_halted). Check both, preferring whichever is actually set,
    so the real question reaches the user instead of the generic fallback.

    ROUTING BACK ON RESUME: clarify is shared by two callers (classify and
    plan), and previously had a single fixed edge to END regardless of
    which one sent it here or whether this was the halt-pass or the
    resume-pass — so answering a clarification never re-entered the
    pipeline at all; the graph just ended with clarify_node's own
    (non-substantive) return dict as the entire final_state, which is
    exactly why chat showed the "(no textual output produced this turn)"
    fallback with no error. We infer which node dispatched here from
    which field was populated on the way in (same discriminator the
    docstring above already uses for picking `question`) and carry that
    forward as "clarification_return_to", so route_after_clarify (see
    build_graph) can send execution back to re-run classify or plan with
    the clarified input folded in, instead of always ending the graph.

    HARD ROUND CAP: prompt-level guidance (config/prompts/classify.yaml)
    asks the model to stop re-clarifying once the person has answered and
    to treat "stop asking" as a signal — but that's a soft guardrail, and
    classify is a fresh LLM call each time, so a model that keeps finding
    a new angle to ask about (observed: three classify->clarify rounds in
    a row for a single run, ~80-100s each, before any real output) isn't
    actually prevented by prompting alone. clarification_rounds counts how
    many times THIS run has hit clarify; once it reaches CLARIFY_ROUND_CAP
    we stop honouring a further clarification_question/needed from
    classify/plan entirely and force route_after_clarify to send this to
    describe instead — describe always produces a textual answer from
    whatever's in normalised_input (which by now includes the full
    Q&A history), so the person gets a best-effort result instead of a
    fourth question.
    """
    import json
    from pathlib import Path
    from langgraph.types import interrupt

    CLARIFY_ROUND_CAP = 2

    question = (
        state.get("clarification_question")
        or state.get("clarification_needed")
        or "Clarification needed."
    )
    # classify_node populates clarification_question (TaskClassification);
    # plan_node populates clarification_needed (PlanSpec). Whichever is
    # actually set tells us which node to re-run on resume.
    return_to = "plan" if state.get("clarification_needed") else "classify"

    rounds = int(state.get("clarification_rounds") or 0) + 1

    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if rounds > CLARIFY_ROUND_CAP:
        # Cap hit — don't halt again. Fold a note into the input so
        # downstream nodes (describe, in particular) know clarification
        # was cut short rather than never attempted, and proceed with
        # best judgment instead of asking a question nobody will answer
        # differently than before.
        existing_input = state.get("normalised_input") or state.get("raw_text_input", "")
        return {
            "pipeline_halted":         False,
            "clarification_needed":    None,
            "clarification_question":  None,
            "clarification_rounds":    rounds,
            "clarification_return_to": "describe",
            "normalised_input": (
                f"{existing_input}\n\n"
                f"(Clarification round limit reached — proceed with your "
                f"best judgment on the remaining ambiguity rather than "
                f"asking again: {question})"
            ),
        }

    (run_dir / "clarification.json").write_text(
        json.dumps({"question": question}, indent=2), encoding="utf-8"
    )

    # Execution suspends here. When resumed via Command(resume=answer),
    # `interrupt()` returns `answer` and the function continues below.
    answer = interrupt(question)

    # We're back — the person answered. Clear the halt markers and fold
    # their answer into normalised_input so downstream nodes (plan/draft/
    # etc.) see it as part of the task. /clarify already removed
    # clarification.json for us before resuming.
    existing_input = state.get("normalised_input") or state.get("raw_text_input", "")
    return {
        "pipeline_halted":         False,
        "clarification_needed":    None,
        "clarification_question":  None,
        "clarification_rounds":    rounds,
        "clarification_return_to": return_to,
        "normalised_input":        f"{existing_input}\n\nClarification: {question}\nAnswer: {answer}",
    }


def route_after_clarify(state) -> str:
    """
    Where to send execution once clarify_node resumes with an answer.
    clarify is a shared halt point (both classify and plan can route
    here) with a single fixed exit edge to END previously — meaning an
    answered clarification never re-entered the graph at all, it just
    ended with clarify_node's own return dict as the whole final_state.
    clarification_return_to (set by clarify_node above — either the
    original caller, or "describe" once CLARIFY_ROUND_CAP is hit) tells
    us where to send execution so the clarified input actually gets used.
    """
    return state.get("clarification_return_to") or "classify"



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
    from pipeline.attachments import compose_input_with_attachments

    run_dir     = state["run_dir"]
    plan        = state.get("plan_spec")
    run_uuid    = state["run_uuid"]
    attachments = state.get("attachments") or []

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
        # Fold in the parent run's attachments (e.g. a schema file, an
        # existing source file to extend) using the same fencing/truncation
        # the main pipeline uses — previously sub-specs never saw attached
        # file content at all, only whatever text the planner distilled
        # into component.responsibility/model_dump_json, which loses any
        # concrete detail (exact column names, exact function signatures)
        # the model would need to implement the component correctly.
        task_input = compose_input_with_attachments(task_input, attachments)

        # Run sub-spec pipeline
        log.info(
            "Running sub-spec %s for component '%s' (%d attachment(s))",
            sub_uuid[:8], component.name, len(attachments),
        )
        sub_result = _run_sub_spec(
            sub_uuid    = sub_uuid,
            sub_dir     = str(sub_dir),
            task_input  = task_input,
            parent_uuid = run_uuid,
            attachments = attachments,
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
    sub_uuid:    str,
    sub_dir:     str,
    task_input:  str,
    parent_uuid: str,
    attachments: Optional[list[dict]] = None,
) -> bool:
    """
    Run a complete short-mode pipeline for a single sub-spec component.
    Returns True if the sub-spec completed successfully.
    """
    import json
    from pathlib import Path

    # Initial state for sub-spec run. task_input already has attachment
    # content folded in by the caller (sub_spec_runner_node), but
    # "attachments" is also carried as its own field — distiller_node's
    # _scrub_attachment_references reads state.get("attachments") directly
    # to redact filenames/content from any lesson this sub-spec distills,
    # and previously got an empty list here regardless of what the parent
    # run had attached.
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
        "attachments":     attachments or [],
        "decompose":       False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
    }

    try:
        app, callbacks = get_graph()
        config = {"configurable": {"thread_id": sub_uuid}}
        if callbacks:
            config["callbacks"] = callbacks
        final_state = app.invoke(initial_state, config=config)
        return not final_state.get("pipeline_failed", True)
    except Exception as e:
        log.error("Sub-spec %s failed: %s", sub_uuid[:8], e)
        return False


# ── Truncation-retry node wrapper ─────────────────────────────────────────────
#
# Every node below calls call_role/call_model (directly or, in describe's
# case, its own equivalent — see nodes/describe.py) and none of them catch
# clients.llm.TruncatedOutputError specifically. Left uncaught, it
# propagates straight out of app_graph.invoke() into server.py's outer
# `except Exception`, which marks the whole run status="error" and
# re-raises — no failure_reason, no chat message, no partial content, and
# critically, no way to retry just that one stage: the entire run is
# already reported as dead by the time anyone could ask for a retry.
#
# UPDATED under the pipeline-profile/escalation design (see
# docs/pipeline-profile-escalation-design.md §2.3/§2.5/§4 item 4):
# clients/llm.py's call_role() now tries ONE automatic escalation step (next
# model on the stage's escalation_ladders entry, config/models.yaml) before
# a TruncatedOutputError can even reach this wrapper. So a
# TruncatedOutputError arriving HERE means the ladder is either already
# exhausted for that stage or the stage has no ladder at all (e.g. it's
# already on the biggest model owned, or deliberately excluded like
# search_query) — this wrapper is now the FALLBACK path once escalation has
# nothing left to try, not the first line of defence. It's kept deliberately
# (not removed) per §4 item 4: a person may genuinely want a higher hard cap
# than any ladder step would apply on its own, independent of model choice.
#
# This wrapper is applied to every node (clarify_node excepted — it manages
# its own interrupt() call directly, see below) at registration time in
# get_graph(), so no individual node file needs to change. It reuses
# exactly the same interrupt()/Command(resume=...)/MemorySaver checkpoint
# mechanism clarify_node already relies on:
#
#   1. Run the wrapped node normally.
#   2. If it raises TruncatedOutputError, DON'T let it propagate. Instead
#      call interrupt() with the truncation details (stage, cap,
#      tokens_out, thinking_block, partial_answer). This pauses the graph
#      at exactly this node — invoke() returns early with "__interrupt__"
#      in the result, same as a clarification halt — and the checkpoint
#      remembers we're mid-way through THIS node specifically.
#   3. server.py's retry endpoint (POST /run/{run_uuid}/retry-truncated)
#      resumes with Command(resume={"budget_tokens": <new cap>}).
#      interrupt() returns that dict here.
#   4. We set PIPELINE_STEP_OUTPUT_CAP_OVERRIDE for the duration of ONE
#      re-invocation of the SAME node function (not the whole graph from
#      classify/plan — just this one node runs again), then clear it.
#   5. If it truncates AGAIN, we interrupt() again with the new details —
#      so a person can bump the cap more than once if needed. If it
#      succeeds, its normal return dict flows on to the next node exactly
#      as if it had never truncated.
#   6. Before interrupting (step 2), also write <node_name>_partial.json
#      alongside truncated.json — see design doc §4 item 6. This is a
#      general fix, not classify-specific: classify_node/plan_node's own
#      TruncatedOutputError handlers just `raise` with no post-loop write
#      of their own to piggyback on, so exc.thinking_block/exc.partial_answer
#      were reaching this wrapper but nothing downstream ever persisted
#      them. Fixed once, here, since the wrapper already has both values
#      and already knows run_dir — every wrapped node gets a partial-save
#      for free with no per-node-file changes needed.
#
# A node can be resumed into more than once in a row (repeated truncation),
# which is why step 5 loops rather than giving up after one retry attempt.
def _wrap_node_for_truncation_retry(node_fn, node_name: str):
    from clients.llm import TruncatedOutputError
    from langgraph.types import interrupt
    import functools
    import json as _json
    from pathlib import Path as _Path

    @functools.wraps(node_fn)
    def wrapped(state):
        while True:
            try:
                return node_fn(state)
            except TruncatedOutputError as exc:
                payload = {
                    "type":            "truncated",
                    "node":            node_name,
                    "stage":           exc.stage,
                    "cap":             exc.cap,
                    "tokens_out":      exc.tokens_out,
                    "thinking_block":  exc.thinking_block,
                    "partial_answer":  exc.partial_answer,
                }
                # Mirror clarify_node's clarification.json convention:
                # server.py's _run_status()/GET /run read a sentinel file
                # on disk rather than inspecting LangGraph's
                # final_state["__interrupt__"] payload directly (that
                # value is only available at the moment invoke() returns
                # — a later GET /run has no access to it). Written to the
                # per-turn run_dir this node is actually executing under
                # (state["run_dir"] — the turn-specific directory for a
                # chat follow-up, see server.py's get_chat_turn_dir), NOT
                # the top-level run dir, so it's associated with the right
                # chat turn rather than always looking like run 1.
                try:
                    run_dir = _Path(state.get("run_dir", ""))
                    if run_dir:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / "truncated.json").write_text(
                            _json.dumps(payload, indent=2), encoding="utf-8"
                        )
                        # ── Generic partial-output save (design doc §4 item 6) ──
                        # Bug: classify_node/plan_node's own TruncatedOutputError
                        # handlers just `raise` — neither node has a post-loop
                        # write of its own to piggyback a partial-save onto, so
                        # exc.thinking_block/exc.partial_answer (already
                        # extracted by clients/llm.py before the exception was
                        # raised) were reaching this wrapper but nothing was
                        # ever writing them anywhere. This IS a general pattern,
                        # not classify-specific — any current or future wrapped
                        # node with a similar retry loop and no post-loop write
                        # hits the identical gap. Fixed here, once, in the
                        # wrapper itself (which already has exc.thinking_block/
                        # exc.partial_answer and already knows run_dir) rather
                        # than in each node file — every wrapped node gets this
                        # for free with no per-node changes required, matching
                        # this wrapper's existing "no individual node file
                        # needs to change" design.
                        try:
                            (run_dir / f"{node_name}_partial.json").write_text(
                                _json.dumps(
                                    {
                                        "node":           node_name,
                                        "stage":          exc.stage,
                                        "cap":             exc.cap,
                                        "tokens_out":      exc.tokens_out,
                                        "thinking_block":  exc.thinking_block,
                                        "partial_answer":  exc.partial_answer,
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                        except Exception:
                            log.warning(
                                "Failed to write %s_partial.json for node '%s'",
                                node_name, node_name,
                            )
                except Exception:
                    log.warning("Failed to write truncated.json sentinel for node '%s'", node_name)

                resume_value = interrupt(payload)

                # Resumed — clear the sentinels so status reporting doesn't
                # keep showing "truncated" (or stale partial content) after
                # a successful retry.
                try:
                    run_dir = _Path(state.get("run_dir", ""))
                    sentinel = run_dir / "truncated.json"
                    if sentinel.exists():
                        sentinel.unlink()
                    partial_file = run_dir / f"{node_name}_partial.json"
                    if partial_file.exists():
                        partial_file.unlink()
                except Exception:
                    pass
                new_cap = None
                if isinstance(resume_value, dict):
                    new_cap = resume_value.get("budget_tokens") or resume_value.get("output_cap")
                if not new_cap:
                    # Defensive fallback — should not normally happen, since
                    # the retry endpoint always supplies a cap. Double the
                    # cap that just failed rather than retrying with the
                    # exact same (guaranteed-to-fail-again) value.
                    new_cap = max(int(exc.cap) * 2, int(exc.cap) + 256)

                with env_lock:
                    prev = os.environ.get("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE")
                    os.environ["PIPELINE_STEP_OUTPUT_CAP_OVERRIDE"] = str(int(new_cap))
                try:
                    log.info(
                        "Retrying node '%s' (stage=%s) with output_cap=%d "
                        "after truncation at cap=%s",
                        node_name, exc.stage, new_cap, exc.cap,
                    )
                    # Loop back to `try: return node_fn(state)` above with
                    # the override now in place for this one call.
                    continue
                finally:
                    with env_lock:
                        if prev is None:
                            os.environ.pop("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE", None)
                        else:
                            os.environ["PIPELINE_STEP_OUTPUT_CAP_OVERRIDE"] = prev

    return wrapped


# env_lock (imported from clients.llm) guards PIPELINE_STEP_OUTPUT_CAP_OVERRIDE
# the same single process-wide lock api/server.py uses for
# PIPELINE_ULTRA/PIPELINE_FORCE_SHORT/PIPELINE_NO_ENSEMBLE — see clients/llm.py's
# env_lock docstring for why this needs to be ONE shared lock object rather
# than a separate Lock() per module.
import os
from clients.llm import env_lock


# ── Graph construction ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    """
    Build and compile the LangGraph StateGraph.
    Cached — compiled once per process.
    """
    builder = StateGraph(PipelineState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    # Every node except clarify is wrapped with the truncation-retry
    # handler above — clarify_node manages its own interrupt() call and
    # must not be double-wrapped (see _wrap_node_for_truncation_retry's
    # docstring). Wrapping here, at registration, means no individual node
    # file needs to import or call anything new to get retry support.
    def _n(name, fn):
        return _wrap_node_for_truncation_retry(fn, name)

    builder.add_node("vision_decode",   _n("vision_decode", vision_decode_node))
    builder.add_node("gatekeeper",      _n("gatekeeper", gatekeeper_node))
    builder.add_node("classify",        _n("classify", classify_node))
    builder.add_node("ideation",        _n("ideation", ideation_node))
    builder.add_node("plan",            _n("plan", plan_node))
    builder.add_node("draft",           _n("draft", draft_node))
    builder.add_node("draft_short",     _n("draft_short", draft_short_node))
    builder.add_node("appraise",        _n("appraise", appraise_node))
    builder.add_node("bugfix",          _n("bugfix", bugfix_node))
    builder.add_node("critic_a",        _n("critic_a", critic_a_node))
    builder.add_node("critic_b",        _n("critic_b", critic_b_node))
    builder.add_node("synthesise",      _n("synthesise", synthesise_node))
    builder.add_node("validate",        _n("validate", validate_node))
    builder.add_node("sub_spec_runner", _n("sub_spec_runner", sub_spec_runner_node))
    builder.add_node("final_validate",  _n("final_validate", final_validate_node))
    builder.add_node("describe",        _n("describe", describe_node))

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

    # ── Plan → draft / draft_short / sub_spec_runner / clarify ──────────────
    # "clarify" here covers plan_node's own halt-for-clarification path
    # (PlanSpec.confidence=low), separate from classify's confidence check.
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "draft":           "draft",
            "draft_short":     "draft_short",
            "sub_spec_runner": "sub_spec_runner",
            "clarify":         "clarify",
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
    # "distiller" here routes to "finalize" first, not the distiller node
    # directly — see the finalize/distiller wiring comment below for why.
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "distiller": "finalize",
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

    # ── Final validate → finalize → distiller ────────────────────────────────
    # We change this routing logic. Instead of going to END, it goes to finalize.
    builder.add_conditional_edges(
        "final_validate",
        route_after_final_validate,  # Ensure this function returns "distiller" on success instead of END
        {"distiller": "finalize", "end": END},
    )

    # ── Finalize → Distiller ──────────────────────────────────────────────────
    # distiller_node's only job is self-improvement bookkeeping (extracting
    # a lesson into the lesson store — see nodes/distiller.py's module
    # docstring); it never builds or writes a user-facing answer. Nothing
    # else in this graph set final_output_path either, so EVERY successful
    # run reached END with it unset, and server.py's _extract_reply_text()
    # fell through to a raw str(fixed_output)/str(draft_output) Pydantic
    # repr as the chat reply (e.g. "applied_fixes=[AppliedFix(...)]
    # self_identified_issues=[...] overall_quality='...'") instead of any
    # real answer — on every run, not as an edge case. finalize_node
    # assembles the actual answer from the (already bugfix-applied)
    # draft_output and writes final.json / final_output_path before
    # distiller does its lesson-extraction pass. route_after_validate and
    # route_after_final_validate's conditional-edge maps above still say
    # "distiller" as the map KEY (matching what those router functions
    # literally return — unchanged) but now point at "finalize" as the
    # target node, so no changes were needed in routers.py itself.
    from nodes.finalize import finalize_node
    builder.add_node("finalize", _n("finalize", finalize_node))
    builder.add_edge("finalize", "distiller")

    # ── Add the new Distiller Node ───────────────────────────────────────────
    from nodes.distiller import distiller_node
    builder.add_node("distiller", _n("distiller", distiller_node))
    builder.add_edge("distiller", END)

    builder.add_edge("describe", END)
    builder.add_node("clarify", clarify_node)
    # Was a fixed edge to END, which fired identically on the halt-pass
    # (correct — invoke() should return early via __interrupt__ before
    # this edge is even evaluated) and on the resume-pass (wrong — an
    # answered clarification needs to re-run classify or plan with the
    # clarified input, not end the graph with clarify_node's own minimal
    # return dict as the final state). See route_after_clarify above.
    # "describe" is the CLARIFY_ROUND_CAP escape hatch — once clarify_node
    # decides not to halt again, it routes here instead so the run still
    # produces a real textual answer rather than looping clarify forever.
    builder.add_conditional_edges("clarify", route_after_clarify, {
        "classify": "classify",
        "plan":     "plan",
        "describe": "describe",
    })
    # ── Gatekeeper always flows to Validate ──────────────────────────────────
    builder.add_edge("gatekeeper", "validate")

    # ── Compile with Langfuse callback and a persistent checkpointer ─────────
    # MemorySaver was in-process-memory only: a clarification pause survives
    # exactly as long as this Python process does. Any server restart,
    # --reload, or crash between a clarify interrupt() and the person
    # answering it wipes the checkpoint out from under Command(resume=...),
    # which then re-enters the graph at set_conditional_entry_point with
    # essentially empty state — route_after_input routes to "classify" same
    # as always, but classify_node's state["run_dir"] KeyErrors because
    # there's no checkpoint left to have carried it forward.
    #
    # SqliteSaver persists to disk, so a resume works regardless of process
    # lifetime in between. The connection is opened once here and kept open
    # for the process lifetime (not via `with`, which would close it as
    # soon as get_graph() returns) since get_graph() is @lru_cache'd and
    # called again later — by background pipeline threads and by every
    # resume/replan endpoint — long after this function itself returns.
    # check_same_thread=False because those callers run in different
    # threads than whichever one first triggered compilation.
    callbacks = _build_callbacks()
    db_path = os.environ.get(
        "PIPELINE_CHECKPOINT_DB",
        str(Path(__file__).parent.parent / "runs" / "checkpoints.sqlite"),
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    app = builder.compile(checkpointer=memory)

    log.info("Pipeline graph compiled (%d nodes)", len(builder.nodes))
    return app, callbacks


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