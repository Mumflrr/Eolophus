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
    get_model_config, load_prompt,
    _write_thinking_log, _extract_thinking,
    _log_stage_entry, _get_thinking_budget,
)

log = logging.getLogger(__name__)


def describe_node(state: dict) -> dict:
    """
    Direct answer node — no planning, no drafting, no schema overhead.
    Returns plain text output via the normal output path.
    """
    run_dir = state["run_dir"]
    task    = state.get("normalised_input") or state.get("raw_text_input", "")

    cfg      = get_model_config("9b")
    base_url = cfg["base_url"]
    model_id = cfg["model_id"]

    # Load system prompt from YAML
    prompt_def  = load_prompt("describe")
    system_text = prompt_def.get("system", (
        "You are a knowledgeable assistant. Answer directly and clearly."
    ))

    # Budget from routing.yaml
    budget = _get_thinking_budget("describe")

    try:
        ensure_model_loaded("9b")
    except Exception as e:
        log.warning("model_manager failed: %s — assuming 9B already running", e)

    raw_client = OpenAI(
        base_url=base_url, api_key="local", timeout=300.0, max_retries=0
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
        extra_body = {
            "thinking": {"type": "enabled", "budget_tokens": budget}
        } if budget > 0 else {
            "thinking": {"type": "disabled"}
        },
    )

    raw_content = resp.choices[0].message.content or ""
    usage       = resp.usage
    elapsed_ms  = (time.perf_counter() - start_ts) * 1000

    thinking_block, answer, _ = _extract_thinking(raw_content)
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
        elapsed_ms, "ok", 0,
    )

    log.info("[describe] answered in %.0fms", elapsed_ms)

    output_path = str(Path(run_dir) / "final.json")
    Path(output_path).write_text(
        json.dumps({"answer": answer, "task_type": "describe"}, indent=2),
        encoding="utf-8"
    )

    return {
        "final_output_path": output_path,
        "pipeline_complete": True,
        "pipeline_failed":   False,
    }
