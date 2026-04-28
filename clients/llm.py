"""
clients/llm.py — single wrapper for all model calls.

Every node calls call_model() — nothing else directly instantiates
an Instructor client or OpenAI client.

Responsibilities:
  - Load model config from models.yaml
  - Apply NoWait logit bias for 35B planning calls
  - Apply LLMLingua-2 compression on eligible content
  - Handle thinking mode toggle and budget_tokens
  - Capture thinking output to log file
  - Extract <confidence> tag from thinking output
  - Retry via Instructor on malformed structured output
  - Emit structured stage log entry on every call
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import instructor
import yaml
from openai import OpenAI
from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Config loading ─────────────────────────────────────────────────────────────

_config_cache: dict = {}

def _load_config() -> dict:
    if _config_cache:
        return _config_cache
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    _config_cache.update(data)
    return _config_cache


def _get_thinking_budget(stage: str, complexity: str = "moderate") -> int:
    """
    Get the thinking token budget for a stage from routing.yaml.
    Uses complexity-aware nested config: thinking_budgets.<stage>.<complexity>.
    Falls back to 2048 if not configured.
    """
    import yaml as _yaml
    try:
        rp = Path(__file__).parent.parent / "config" / "routing.yaml"
        with open(rp) as f:
            rcfg = _yaml.safe_load(f)
        stage_cfg = rcfg.get("thinking_budgets", {}).get(stage, {})
        if isinstance(stage_cfg, dict):
            return stage_cfg.get(complexity, stage_cfg.get("moderate", 2048))
        return int(stage_cfg) if stage_cfg else 2048
    except Exception:
        return 2048


def _get_http_timeout() -> float:
    """Read HTTP timeout from routing.yaml. Default 7200s (2 hours)."""
    import yaml as _yaml
    try:
        rp = Path(__file__).parent.parent / "config" / "routing.yaml"
        with open(rp) as f:
            rcfg = _yaml.safe_load(f)
        return float(rcfg.get("http", {}).get("timeout_seconds", 7200))
    except Exception:
        return 7200.0


def get_model_config(model_id: str) -> dict:
    """Return the config block for a model_id (e.g. '9b', '35b')."""
    cfg = _load_config()
    if model_id not in cfg["models"]:
        raise ValueError(f"Unknown model_id '{model_id}'. Check config/models.yaml.")
    return cfg["models"][model_id]


def resolve_role(role: str) -> str:
    """Resolve a role name to a model_id via config/models.yaml roles mapping."""
    cfg = _load_config()
    if role not in cfg["roles"]:
        raise ValueError(f"Unknown role '{role}'. Check config/models.yaml roles section.")
    return cfg["roles"][role]


# ── Thinking capture ───────────────────────────────────────────────────────────

_THINK_OPEN  = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_CONFIDENCE  = re.compile(r"<confidence>(.*?)</confidence>", re.IGNORECASE | re.DOTALL)


def _extract_thinking(raw_content: str) -> tuple[str, str, Optional[str]]:
    """
    Split raw model output into (thinking_block, answer, confidence_signal).

    Models using thinking format produce:
      <think>...reasoning...</think>
      ...final answer...

    Returns:
      thinking_block     — content inside <think>...</think> (empty string if none)
      answer             — content after </think> (or full content if no think tags)
      confidence_signal  — content of <confidence>...</confidence> if present in thinking
    """
    think_match = re.search(r"<think>(.*?)</think>(.*)", raw_content, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking_block = think_match.group(1).strip()
        answer         = think_match.group(2).strip()
    else:
        thinking_block = ""
        answer         = raw_content.strip()

    confidence_signal = None
    if thinking_block:
        conf_match = _CONFIDENCE.search(thinking_block)
        if conf_match:
            confidence_signal = conf_match.group(1).strip()

    return thinking_block, answer, confidence_signal


def _write_thinking_log(run_dir: str, stage: str, thinking_block: str) -> None:
    """Write thinking output to {run_dir}/{stage}_thinking.log"""
    if not thinking_block:
        return
    log_path = Path(run_dir) / f"{stage}_thinking.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        f.write(thinking_block)
        f.write("\n\n")


# ── NoWait logit bias ─────────────────────────────────────────────────────────

def _build_logit_bias(model_cfg: dict) -> Optional[dict[str, float]]:
    """
    Build logit_bias dict for NoWait suppression if configured.
    Token IDs are model-specific and must be populated in models.yaml
    via tools/derive_nowait_tokens.py.
    Returns None if not configured (no-op).
    """
    nowait = model_cfg.get("nowait_tokens", {})
    if not nowait or "_note" in nowait:
        return None
    return {str(k): -100.0 for k in nowait.keys()}


# ── LLMLingua-2 compression ───────────────────────────────────────────────────

_lingua_compressor = None

def _get_compressor():
    global _lingua_compressor
    if _lingua_compressor is None:
        try:
            from llmlingua import PromptCompressor
            _lingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
            log.info("LLMLingua-2 compressor initialised")
        except ImportError:
            log.warning("llmlingua not installed — compression disabled. pip install llmlingua")
            _lingua_compressor = "unavailable"
    return _lingua_compressor if _lingua_compressor != "unavailable" else None


def compress_text(
    text: str,
    ratio: float = 0.5,
    min_tokens: int = 200,
) -> str:
    """
    Compress text using LLMLingua-2 if available and content is long enough.
    Never compress code — callers are responsible for separating code from prose.

    Args:
        text:       Text to compress (prose only, not code)
        ratio:      Target compression ratio (0.5 = keep 50% of tokens)
        min_tokens: Skip compression if content is below this token estimate

    Returns:
        Compressed text, or original text if compression unavailable/skipped.
    """
    # Rough token estimate: 1 token ≈ 4 chars
    estimated_tokens = len(text) / 4
    if estimated_tokens < min_tokens:
        return text

    compressor = _get_compressor()
    if compressor is None:
        return text

    try:
        result = compressor.compress_prompt(
            text,
            rate=ratio,
            force_tokens=["\n"],
        )
        compressed = result.get("compressed_prompt", text)
        log.debug(
            "LLMLingua-2: %.0f -> %.0f tokens (%.1f%% reduction)",
            estimated_tokens,
            len(compressed) / 4,
            (1 - len(compressed) / len(text)) * 100,
        )
        return compressed
    except Exception as e:
        log.warning("LLMLingua-2 compression failed: %s — using original", e)
        return text


# ── Stage logging ─────────────────────────────────────────────────────────────

def _log_stage_entry(
    run_dir:    str,
    stage:      str,
    model_name: str,
    prompt_hash:str,
    tokens_in:  int,
    tokens_out: int,
    latency_ms: float,
    status:     str,
    retries:    int = 0,
) -> None:
    """Append one line to {run_dir}/stages.log"""
    log_path = Path(run_dir) / "stages.log"
    entry = {
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage":      stage,
        "model":      model_name,
        "prompt_hash":prompt_hash,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
        "status":     status,
        "retries":    retries,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Streaming completion helper ───────────────────────────────────────────────

def _stream_completion(
    client,
    model_id:   str,
    messages:   list,
    temp:       float,
    top_p:      float,
    extra_body,
    stage:      str,
):
    """
    Stream a completion, logging progress every 100 tokens.
    Returns (full_content: str, usage).
    Prevents the pipeline appearing hung during long think blocks.
    """
    chunks     = []
    token_count= 0
    in_think   = False
    think_toks = 0
    usage      = None
    last_log   = 0

    try:
        stream = client.chat.completions.create(
            model      = model_id,
            messages   = messages,
            temperature= temp,
            top_p      = top_p,
            extra_body = extra_body,
            stream     = True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                text = delta.content
                chunks.append(text)
                token_count += 1

                # Track think block
                combined = "".join(chunks)
                if "<think>" in combined and not in_think:
                    in_think = True
                if in_think and "</think>" not in combined:
                    think_toks += 1

                # Log progress every 100 tokens
                if token_count - last_log >= 100:
                    if in_think and "</think>" not in combined:
                        log.debug("[%s] thinking... %d tokens", stage, think_toks)
                    else:
                        log.debug("[%s] generating... %d tokens", stage, token_count)
                    last_log = token_count

            # Capture usage from final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

    except Exception as e:
        # Fall back to non-streaming if streaming fails
        log.warning("[%s] streaming failed (%s), falling back to non-streaming", stage, e)
        resp = client.chat.completions.create(
            model      = model_id,
            messages   = messages,
            temperature= temp,
            top_p      = top_p,
            extra_body = extra_body,
        )
        return resp.choices[0].message.content or "", resp.usage

    full_content = "".join(chunks)
    log.debug("[%s] completed: %d total tokens", stage, token_count)
    return full_content, usage


# ── Core call function ────────────────────────────────────────────────────────

def call_model(
    model_id:        str,
    messages:        list[dict],
    response_schema: Type[T],
    stage:           str,
    run_dir:         str,
    thinking:        Optional[bool]  = None,
    budget_tokens:   Optional[int]   = None,
    compress_system: bool            = False,
    compress_ratio:  float           = 0.5,
    max_retries:     int             = 3,
    skip_nowait:     bool            = False,
) -> T:
    """
    Make a structured model call via Instructor.

    Args:
        model_id:        Model identifier from models.yaml (e.g. '9b', '35b')
                         OR a role name resolved via resolve_role() first.
        messages:        List of {"role": ..., "content": ...} dicts.
        response_schema: Pydantic model class to parse the response into.
        stage:           Pipeline stage name for logging (e.g. 'plan', 'draft').
        run_dir:         Path to the current run directory for log files.
        thinking:        Override thinking mode. None = use model default.
        budget_tokens:   Override thinking budget. None = use model default.
        compress_system: If True, apply LLMLingua-2 to system message content.
        compress_ratio:  Compression ratio if compress_system is True.
        max_retries:     Instructor retry attempts on malformed output.

    Returns:
        Populated instance of response_schema.

    Raises:
        instructor.exceptions.InstructorRetryException: if all retries exhausted
        ValueError: if model_id is unknown
    """
    cfg        = get_model_config(model_id)
    base_url   = cfg["base_url"]
    model_name = cfg["name"]
    temp       = cfg.get("temperature", 0.6)
    top_p      = cfg.get("top_p", 0.95)

    # Ensure the model server is running before making the call.
    # On single-GPU setups this may stop the previous model first.
    try:
        from clients.model_manager import ensure_model_loaded
        ensure_model_loaded(model_id)
    except Exception as e:
        log.warning("model_manager.ensure_model_loaded failed: %s — assuming server already running", e)

    # Resolve thinking settings
    thinking_default = cfg.get("thinking", {}).get("default_on", False)
    use_thinking     = thinking if thinking is not None else thinking_default
    
    if budget_tokens is not None:
        tok_budget = budget_tokens
    else:
        # Apply the stage-specific budget from routing.yaml. 
        # This overrides models.yaml defaults!
        tok_budget = _get_thinking_budget(stage)


    # Apply LLMLingua-2 compression to system message if requested
    if compress_system:
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = compress_text(
                    msg["content"], ratio=compress_ratio
                )
                break

    # Build prompt hash (for logging — not full content)
    prompt_str  = json.dumps(messages, sort_keys=True)
    prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()[:12]

    # Build extra_body for thinking mode and logit bias
    # Always send thinking parameter explicitly — omitting it lets the model
    # decide, which causes Qwen3.5 to think by default on every request.
    extra_body: dict[str, Any] = {}
    if use_thinking:
        extra_body["thinking"] = {
            "type":          "enabled",
            "budget_tokens":  tok_budget,
        }
    else:
        extra_body["thinking"] = {"type": "disabled"}
    logit_bias = _build_logit_bias(cfg) if not skip_nowait else None
    if logit_bias:
        extra_body["logit_bias"] = logit_bias

    # Instructor client — long timeout to avoid retry noise during 35B generation
    _http_timeout = _get_http_timeout()
    raw_client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=1200.0)
    client     = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    retries_used = 0
    start_ts     = time.perf_counter()

    try:
        # Stream the response so we can log progress and capture thinking.
        # This prevents the pipeline appearing hung during long think blocks.
        raw_content, usage = _stream_completion(
            raw_client, cfg["model_id"], messages, temp, top_p,
            extra_body if extra_body else None, stage
        )

        # Extract thinking block and confidence signal
        thinking_block, answer, confidence = _extract_thinking(raw_content)

        # Write thinking log
        if thinking_block:
            _write_thinking_log(run_dir, stage, thinking_block)

        # Now parse the answer portion via Instructor for schema validation
        # We do this by creating a synthetic completion and patching via Instructor
        # Alternatively: re-call with the answer as assistant message and ask to format.
        # Simplest: use instructor.from_openai with the answer text directly.
        try:
            result: T = response_schema.model_validate_json(answer)
            retries_used = 0
        except Exception:
            # Instructor retry: ask model to produce a populated INSTANCE not a schema.
            # Critical: show field names only, not the full JSON Schema definition.
            # Sending model_json_schema() causes the model to echo the schema back.
            field_names = list(response_schema.model_fields.keys())
            fields_hint = ", ".join(f'"{f}": <value>' for f in field_names[:6])
            result, completion = client.chat.completions.create_with_completion(
                model    = cfg["model_id"],
                messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user",      "content": (
                        f"Your previous response could not be parsed. "
                        f"Respond with a JSON object that is an INSTANCE (filled-in values), "
                        f"NOT a schema definition. "
                        f"Required fields: {field_names}. "
                        f"Example structure: {{{fields_hint}}}. "
                        f"Do not include $defs, properties, or type keys — "
                        f"those are schema keywords, not values."
                    )},
                ],
                response_model = response_schema,
                max_retries    = max_retries,
                temperature    = temp,
            )
            retries_used = max_retries  # approximate

        # Attach confidence signal if the schema has that field
        if confidence and hasattr(result, "confidence_signal"):
            object.__setattr__(result, "confidence_signal", confidence)

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        _log_stage_entry(
            run_dir, stage, model_name, prompt_hash,
            tokens_in, tokens_out, elapsed_ms,
            "ok", retries_used,
        )

        log.info(
            "[%s] %s → %s | %d+%d tok | %.0fms",
            stage, model_name, response_schema.__name__,
            tokens_in, tokens_out, elapsed_ms,
        )

        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        _log_stage_entry(
            run_dir, stage, model_name, prompt_hash,
            0, 0, elapsed_ms, f"error:{type(exc).__name__}", retries_used,
        )
        log.error("[%s] %s call failed: %s", stage, model_name, exc)
        raise


# ── Convenience: call by role ─────────────────────────────────────────────────

def call_role(
    role:            str,
    messages:        list[dict],
    response_schema: Type[T],
    stage:           str,
    run_dir:         str,
    thinking:        False,
    max_retries:     0,
    **kwargs,
) -> T:
    """
    Like call_model() but resolves role → model_id via models.yaml.
    Preferred in node files to avoid hardcoding model IDs.
    """
    model_id = resolve_role(role)
    return call_model(
        model_id, messages, response_schema, stage, run_dir, **kwargs
    )