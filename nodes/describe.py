"""
nodes/describe.py — direct answer node for describe/analysis tasks.

Fires when task_type=describe. Bypasses plan/draft/appraise/bugfix entirely.
Returns plain text — Instructor/structured output intentionally avoided to
eliminate schema overhead for conversational responses.

System prompt lives in config/prompts/describe.yaml.
The raw OpenAI client is used directly; thinking budget from routing.yaml
is read via _get_thinking_budget("describe").

Web search: describe is also where lookup-style questions land ("search for
today's date", "what's the latest FastAPI release") — see classify.yaml. When
state["use_search"] is set, the search_web tool is offered and a small tool
loop runs before the final answer (same mechanics as clients/llm.py's
call_model_with_tools, but plain text out instead of a schema). With
use_search off, this behaves exactly as it did before.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openai import OpenAI
from clients.model_manager import ensure_model_loaded
from clients.llm import (
    get_model_config, load_prompt, resolve_role, resolve_ultra_model,
    _write_thinking_log, _extract_thinking_partial,
    _log_stage_entry, _get_thinking_budget, _get_http_timeout,
    _get_output_token_cap, TruncatedOutputError, ToolCallRecord, _build_logit_bias,
    _build_thinking_extra_body,
)
from clients.tools import (
    SEARCH_HINT, SEARCH_TOOL_SCHEMA, TOOL_IMPLEMENTATIONS, format_search_notes,
)

log = logging.getLogger(__name__)

# Cap on search round-trips before the model is told to answer with what it
# has (tool_choice="none"). Mirrors call_model_with_tools' default.
_MAX_TOOL_ROUNDS = 4


def _resolve_describe_model(state: dict) -> str:
    """
    Same starting-model priority as call_role() (clients/llm.py), reimplemented
    here since describe_node doesn't go through call_role — see module
    docstring for why. Priority: mid-run escalation override > ultra profile
    (ladder final entry) > profile role_overrides > roles: default (9b).
    Previously this was hardcoded to "9b" regardless of profile or prior
    escalation, meaning a describe truncation could only ever retry on 9b
    with a higher output cap (via PIPELINE_STEP_OUTPUT_CAP_OVERRIDE) and
    could never actually escalate to 35b even though escalation_ladders.
    describe = ["35b"] says it should be able to.
    """
    override = (state.get("escalated_models") or {}).get("describe")
    if override:
        return override

    profile = state.get("profile") or state.get("requested_profile")
    if profile == "ultra":
        return resolve_ultra_model("describe")
    if profile:
        try:
            import yaml
            routing_path = Path(__file__).parent.parent / "config" / "routing.yaml"
            with open(routing_path) as f:
                routing_cfg = yaml.safe_load(f) or {}
            overrides = (
                routing_cfg.get("pipeline_profiles", {})
                .get(profile, {})
                .get("role_overrides", {})
            )
            if "describe" in overrides:
                return overrides["describe"]
        except Exception:
            log.warning("Could not read pipeline_profiles for profile '%s' — using roles: default", profile)

    return resolve_role("describe")


def describe_node(state: dict) -> dict:
    """
    Direct answer node — no planning, no drafting, no schema overhead.
    Returns plain text output via the normal output path.

    describe_node builds its own OpenAI client instead of going through
    call_role/call_model (see module docstring), so it's also responsible
    for its own truncation detection. It now raises TruncatedOutputError
    on truncation — same exception call_model raises — rather than
    hand-rolling a separate pipeline_failed/failure_reason path. This
    means describe truncations flow through the exact same generic
    retry wrapper every other node gets (see pipeline/graph.py's
    _wrap_node_for_truncation_retry): the graph pauses via interrupt(),
    the person can retry with a higher cap, and it re-enters this
    function rather than the whole pipeline.

    Previously this only treated truncation as fatal when NO usable
    answer resulted (cap landed mid-thought). A truncation that still
    left a partial answer was silently reported as pipeline_complete —
    but a cut-off answer is exactly the case someone would want to
    retry for the complete version, so it now raises the same way.
    """
    run_dir  = state["run_dir"]
    task     = state.get("normalised_input") or state.get("raw_text_input", "")

    model_id_key = _resolve_describe_model(state)
    cfg      = get_model_config(model_id_key)
    base_url = cfg["base_url"]
    model_id = cfg["model_id"]

    # Load system prompt from YAML
    prompt_def  = load_prompt("describe")
    system_text = prompt_def.get("system", (
        "You are a knowledgeable assistant. Answer directly and clearly."
    ))

    # Budget from routing.yaml
    budget = _get_thinking_budget("describe")
    # Hard cap on completion length — thinking_budgets.describe only guides
    # the thinking phase; without this, a task can generate far past the
    # thinking budget (seen: 21k+ tokens on one run) with only the HTTP
    # timeout to eventually stop it. NOTE: max_tokens caps thinking + answer
    # combined, so this must stay comfortably above budget (currently 512)
    # or the answer itself can get truncated once thinking eats into the cap.
    #
    # The truncation-retry wrapper (graph.py) re-runs this exact node with
    # a higher cap after a TruncatedOutputError, via step_overrides()
    # (clients/llm.py) — read directly here since describe_node doesn't go
    # through call_role/call_model, where it's normally consulted.
    from clients.llm import _step_output_cap_override
    _cap_override = _step_output_cap_override.get()
    max_tokens = _cap_override if _cap_override is not None else _get_output_token_cap("describe")

    try:
        ensure_model_loaded(model_id_key)
    except Exception as e:
        log.warning("model_manager failed: %s — assuming %s already running", e, model_id_key)

    raw_client = OpenAI(
        # Previously hardcoded to timeout=300.0 — same bug as the one fixed
        # in clients/llm.py's call_model(), just duplicated here since
        # describe_node builds its own client instead of going through
        # call_role(). Now respects routing.yaml's http.timeout_seconds
        # (default 7200s) like every other node.
        base_url=base_url, api_key="local", timeout=_get_http_timeout(), max_retries=0
    )

    # Web search is opt-in per run (state["use_search"]). Offered as a real
    # tool, and SEARCH_HINT is put in front of the task so the model is told
    # it can search — a 9B won't reliably call a tool nobody mentioned.
    call_tools = [SEARCH_TOOL_SCHEMA] if state.get("use_search") else None
    user_text  = (SEARCH_HINT + task) if call_tools else task

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]

    # ── Request shape ─────────────────────────────────────────────────────
    # describe builds its own client, so it has to send everything call_model /
    # call_model_with_tools send or it quietly runs with less protection than
    # every other node. It was missing four things:
    #   - top_p               (cfg, default 0.95 — both call paths pass it)
    #   - reasoning_budget    (the parameter llama.cpp reads; the Anthropic-style
    #                          thinking={budget_tokens} key alone is ignored by
    #                          llama.cpp, which is why the "512" here never bound)
    #   - logit_bias NoWait   (models.yaml nowait_tokens; suppresses the "Wait…"
    #                          self-doubt loops a thinking Qwen falls into — the
    #                          reason call_model applies it to every node)
    #   - presence_penalty    (models.yaml presence_penalty; was declared on 35b
    #                          with the comment "prevents infinite loops in <think>
    #                          tags" but never actually read by ANY call site in
    #                          the codebase, so it did nothing even there. NoWait
    #                          above stops a specific self-doubt pattern in
    #                          thinking; this is the general guard against plain
    #                          content-level repetition — e.g. a free-text answer
    #                          that degenerates into repeating "|" forever, which
    #                          NoWait suppression does nothing for)
    # Unlike the JSON-schema nodes, which stop when the object closes, describe
    # is free text — max_tokens is still the last line of defense, but it
    # shouldn't be the only one.
    top_p             = cfg.get("top_p", 0.95)
    presence_penalty  = cfg.get("presence_penalty")
    nowait_bias       = _build_logit_bias(cfg)

    def _request_extra_body(thinking_on: bool) -> dict:
        # The shared builder (clients/llm.py) — the same one call_model and
        # call_model_with_tools use, so the three can't drift apart again.
        body = _build_thinking_extra_body(thinking_on, budget)
        if nowait_bias:
            body["logit_bias"] = nowait_bias
        return body

    tool_history: list[ToolCallRecord] = []
    stats = {"tokens_out": 0, "rounds": 0, "reasoning_chars": 0}

    def _generate(thinking_on: bool):
        """
        One full generation: the tool loop (when search is on), then the answer.
        Returns (raw_content, finish_reason, usage). Called once normally, and
        a second time with thinking disabled if the first ran away in thinking.
        """
        tool_history.clear()                     # a retry starts from a clean slate
        extra_body       = _request_extra_body(thinking_on)
        working_messages = list(messages)        # grows with tool turns; `messages`
                                                 # stays system+user for the prompt hash
        rounds = 0
        while True:
            rounds += 1
            stats["rounds"] += 1
            force_final = bool(call_tools) and rounds > _MAX_TOOL_ROUNDS

            create_kwargs = dict(
                model       = model_id,
                messages    = working_messages,
                temperature = cfg.get("temperature", 0.6),
                top_p       = top_p,
                max_tokens  = max_tokens,   # None = unbounded, matches prior behaviour
                extra_body  = extra_body,
            )
            if presence_penalty is not None:
                create_kwargs["presence_penalty"] = presence_penalty
            if call_tools:
                create_kwargs["tools"]       = call_tools
                create_kwargs["tool_choice"] = "none" if force_final else "auto"

            resp  = raw_client.chat.completions.create(**create_kwargs)
            msg   = resp.choices[0].message
            usage = resp.usage
            if usage:
                stats["tokens_out"] += usage.completion_tokens

            # llama.cpp puts thinking on tool-calling responses in a separate
            # reasoning_content field rather than <think> tags in content (see
            # call_model_with_tools' docstring) — log it per round.
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                stats["reasoning_chars"] += len(reasoning)
                _write_thinking_log(run_dir, "describe", reasoning)

            if call_tools and msg.tool_calls and not force_final:
                log.info("[describe] round %d: model requested %d tool call(s)",
                         rounds, len(msg.tool_calls))
                working_messages.append(msg.model_dump(exclude_none=True))
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        log.warning("[describe] tool call %s had unparseable arguments (%s): %r",
                                    tool_name, e, tc.function.arguments)
                        tool_args = {}

                    impl = TOOL_IMPLEMENTATIONS.get(tool_name)
                    if impl is None:
                        result_text = f"Error: unknown tool '{tool_name}'."
                        log.warning("[describe] model called unregistered tool '%s'", tool_name)
                    else:
                        try:
                            result_text = impl(tool_args)
                        except Exception as e:
                            # Best-effort, same as call_model_with_tools: hand
                            # the error back as the tool result so the model can
                            # answer without it instead of crashing the node.
                            log.warning("[describe] tool '%s' raised: %s", tool_name, e)
                            result_text = f"Error running tool: {e}"

                    tool_history.append(ToolCallRecord(
                        name=tool_name, arguments=tool_args, result=result_text,
                    ))
                    working_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_text,
                    })
                continue   # go again with the tool results in context

            # No tool calls (or a forced final answer) — this is the answer.
            return msg.content or "", resp.choices[0].finish_reason, usage

    start_ts = time.perf_counter()
    raw_content, finish_reason, usage = _generate(thinking_on=budget > 0)
    truncated = finish_reason == "length"

    # ── Runaway-thinking recovery ─────────────────────────────────────────
    # Cap hit with NO answer at all means it never got out of the thinking
    # phase (a "Wait…" loop). Retrying with the same settings and a bigger cap
    # (what the truncation-retry wrapper would offer) just loops for longer, so
    # try once more with thinking off — describe is a direct-answer node, and
    # an unthought answer beats a 16k-token failure. A truncation that DID
    # produce answer text is a genuinely long answer and still raises below.
    thinking_block, answer, _ = _extract_thinking_partial(raw_content)
    if truncated and budget > 0 and not answer.strip():
        log.warning(
            "[describe] hit the %s-token cap without leaving the thinking phase "
            "(%d chars reasoning, %d chars in-content thinking; budget %d was not "
            "enforced) — retrying once with thinking disabled",
            max_tokens, stats["reasoning_chars"], len(thinking_block), budget,
        )
        if thinking_block:
            _write_thinking_log(run_dir, "describe", thinking_block)
        raw_content, finish_reason, usage = _generate(thinking_on=False)
        truncated = finish_reason == "length"

    elapsed_ms  = (time.perf_counter() - start_ts) * 1000

    # Use the truncation-aware extractor (handles an <think> block left
    # open with no closing tag — the common case when the cap lands
    # mid-thought, see clients/llm.py's _extract_thinking_partial) so a
    # cut-off-mid-thought call still surfaces its thinking rather than
    # dumping the unterminated block into `answer`.
    thinking_block, answer, _ = _extract_thinking_partial(raw_content)
    if thinking_block:
        _write_thinking_log(run_dir, "describe", thinking_block)

    import hashlib, json as _json
    prompt_hash = hashlib.sha256(
        _json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()[:12]

    _log_stage_entry(
        run_dir, "describe", cfg["name"], prompt_hash,
        usage.prompt_tokens     if usage else 0,
        stats["tokens_out"],
        elapsed_ms, "truncated" if truncated else "ok", 0,
        think_ratio = round(
            (stats["reasoning_chars"] + len(thinking_block)) /
            max(stats["reasoning_chars"] + len(thinking_block) + len(answer), 1), 3),
    )

    if truncated:
        log.warning(
            "[describe] hit max_tokens cap (%s) — %d chars thinking, "
            "%d chars partial answer",
            max_tokens, len(thinking_block), len(answer),
        )
        # Attempt one ladder step, same rule as call_role's internal
        # escalation walk (clients/llm.py) — mutate state in place so
        # graph.py's _wrap_node_for_truncation_retry's re-entry into this
        # node picks up the escalated model via _resolve_describe_model's
        # escalated_models check above, rather than only ever retrying the
        # SAME model with a bumped output cap (which is what happened here
        # before describe had a model to escalate TO at all).
        from clients.llm import next_escalation_model
        next_model = next_escalation_model("describe", model_id_key)
        if next_model:
            escalated_models = dict(state.get("escalated_models") or {})
            escalation_history = list(state.get("escalation_history") or [])
            escalated_models["describe"] = next_model
            escalation_history.append({
                "stage":      "describe",
                "from_model": model_id_key,
                "to_model":   next_model,
                "trigger":    "truncation",
                "iteration":  state.get("iteration", 0),
            })
            state["escalated_models"]   = escalated_models
            state["escalation_history"] = escalation_history
            log.warning("Describe truncated on '%s' — escalating to '%s' for retry",
                        model_id_key, next_model)
        raise TruncatedOutputError(
            stage=  "describe",
            cap=    max_tokens if max_tokens is not None else (usage.completion_tokens if usage else 0),
            tokens_out=usage.completion_tokens if usage else 0,
            thinking_block=thinking_block,
            partial_answer=answer,
        )

    log.info("[describe] answered in %.0fms (%d round(s), %d search call(s))",
             elapsed_ms, stats["rounds"], len(tool_history))

    output_path = str(Path(run_dir) / "final.json")
    Path(output_path).write_text(
        json.dumps({"answer": answer, "task_type": "describe"}, indent=2),
        encoding="utf-8"
    )

    result = {
        "final_output_path": output_path,
        "pipeline_complete": True,
        "pipeline_failed":   False,
        # Overwritten every pass ("" when nothing was searched) so a chat
        # follow-up on a reused checkpoint can't show a previous turn's results.
        "search_notes":      format_search_notes(tool_history),
    }
    if "escalated_models" in state:
        result["escalated_models"] = state["escalated_models"]
    if "escalation_history" in state:
        result["escalation_history"] = state["escalation_history"]
    return result