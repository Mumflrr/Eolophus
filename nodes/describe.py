"""
nodes/describe.py — direct answer node for describe/analysis tasks.

Fires when task_type=describe. Bypasses plan/draft/appraise/bugfix entirely.
Returns plain text — Instructor/structured output intentionally avoided to
eliminate schema overhead for conversational responses.

System prompt lives in config/prompts/describe.yaml.
The raw OpenAI client is used directly; thinking budget from routing.yaml
is read via _get_thinking_budget("describe").
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
    _get_output_token_cap, TruncatedOutputError,
)

log = logging.getLogger(__name__)


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
    # PIPELINE_STEP_OUTPUT_CAP_OVERRIDE takes precedence when set — this is
    # how the truncation-retry wrapper re-runs this exact node with a
    # higher cap after a TruncatedOutputError (see graph.py). Read via
    # os.environ directly here since describe_node doesn't go through
    # call_role/call_model, which is where that env var is normally
    # consulted (clients/llm.py's call_role).
    import os
    _cap_override = os.environ.get("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE")
    max_tokens = int(_cap_override) if _cap_override is not None else _get_output_token_cap("describe")

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

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": task},
    ]

    start_ts = time.perf_counter()

    resp = raw_client.chat.completions.create(
        model      = model_id,
        messages   = messages,
        temperature= cfg.get("temperature", 0.6),
        max_tokens = max_tokens,   # None = unbounded, matches prior behaviour
        extra_body = {
            "thinking": {"type": "enabled", "budget_tokens": budget}
        } if budget > 0 else {
            "thinking": {"type": "disabled"}
        },
    )

    raw_content = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    usage       = resp.usage
    elapsed_ms  = (time.perf_counter() - start_ts) * 1000
    truncated   = finish_reason == "length"

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
        usage.completion_tokens if usage else 0,
        elapsed_ms, "truncated" if truncated else "ok", 0,
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

    log.info("[describe] answered in %.0fms", elapsed_ms)

    output_path = str(Path(run_dir) / "final.json")
    Path(output_path).write_text(
        json.dumps({"answer": answer, "task_type": "describe"}, indent=2),
        encoding="utf-8"
    )

    result = {
        "final_output_path": output_path,
        "pipeline_complete": True,
        "pipeline_failed":   False,
    }
    if "escalated_models" in state:
        result["escalated_models"] = state["escalated_models"]
    if "escalation_history" in state:
        result["escalation_history"] = state["escalation_history"]
    return result