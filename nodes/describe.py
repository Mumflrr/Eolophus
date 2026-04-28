"""
nodes/describe.py — direct answer node for describe/analysis tasks.

Fires when task_type=describe. Bypasses plan/draft/appraise/bugfix entirely.
The 9B answers directly in thinking mode with no structured schema overhead —
just a plain text response.

Use cases:
  - "describe the most notable features of this FEN: ..."
  - "what is the time complexity of quicksort?"
  - "explain the difference between mutex and semaphore"
  - "analyze this position"
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from openai import OpenAI
from clients.model_manager import ensure_model_loaded
from clients.llm import get_model_config, _write_thinking_log, _extract_thinking, _log_stage_entry

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a knowledgeable assistant. Answer the question directly and clearly. "
    "If the question involves chess, be specific about pieces and squares. "
    "If it involves code or concepts, be precise and concrete."
)


def describe_node(state: dict) -> dict:
    """
    Direct answer node — no planning, no drafting, no schema overhead.
    Returns plain text output via the normal output path.
    """
    run_dir = state["run_dir"]
    task    = state.get("normalised_input") or state.get("raw_text_input", "")

    cfg        = get_model_config("9b")
    base_url   = cfg["base_url"]
    model_id   = cfg["model_id"]

    try:
        ensure_model_loaded("9b")
    except Exception as e:
        log.warning("model_manager failed: %s — assuming 9B already running", e)

    raw_client = OpenAI(base_url=base_url, api_key="local", timeout=300.0, max_retries=0)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": task},
    ]

    start_ts = time.perf_counter()

    resp = raw_client.chat.completions.create(
        model       = model_id,
        messages    = messages,
        temperature = 0.6,
        extra_body  = {"thinking": {"type": "enabled", "budget_tokens": 512}},
    )

    raw_content = resp.choices[0].message.content or ""
    usage       = resp.usage
    elapsed_ms  = (time.perf_counter() - start_ts) * 1000

    thinking_block, answer, _ = _extract_thinking(raw_content)
    if thinking_block:
        _write_thinking_log(run_dir, "describe", thinking_block)

    _log_stage_entry(
        run_dir, "describe", cfg["name"], "direct",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        elapsed_ms, "ok", 0,
    )

    log.info("[describe] answered in %.0fms", elapsed_ms)

    # Write answer as the final output — wrap in a minimal structure
    # so _extract_output() in run.py can find it
    output_path = str(Path(run_dir) / "final.json")
    import json
    Path(output_path).write_text(
        json.dumps({"answer": answer, "task_type": "describe"}, indent=2),
        encoding="utf-8"
    )

    return {
        "final_output_path": output_path,
        "pipeline_complete": True,
        "pipeline_failed":   False,
    }
