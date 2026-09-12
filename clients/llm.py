"""
clients/llm.py — single wrapper for all model calls.

Every node calls call_role() — nothing else directly instantiates
an Instructor client or OpenAI client.

Responsibilities:
  - Load model config from models.yaml
  - Load prompt templates from config/prompts/{role}.yaml
  - Render system + user messages from YAML templates + caller-supplied vars
  - Inject confidence instruction automatically when schema has confidence field
  - Apply NoWait logit bias for 35B planning calls
  - Apply LLMLingua-2 compression on eligible content
  - Handle thinking mode toggle and budget_tokens
  - Capture thinking output to log file
  - Extract <confidence> tag from thinking output and attach to result
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
import threading
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import instructor
import yaml
from openai import OpenAI
from pydantic import BaseModel

# Single process-wide lock guarding mutations of the PIPELINE_STEP_*
# environment variables this module reads (PIPELINE_STEP_BUDGET_OVERRIDE,
# PIPELINE_STEP_OUTPUT_CAP_OVERRIDE) as well as the PIPELINE_NO_ENSEMBLE /
# PIPELINE_ULTRA / PIPELINE_FORCE_SHORT flags api/server.py sets around a
# run. Previously server.py and pipeline/graph.py each defined their own
# threading.Lock() instance under the same name (_env_lock) — since
# os.environ is process-global but each Lock object only excludes callers
# holding THAT SAME instance, two separate locks gave no real mutual
# exclusion between the two modules at all, just each protecting its own
# module's critical section against itself. That happened to be harmless
# only because the two modules mutate different variable names in
# practice — this makes it actually correct rather than accidentally so,
# by having every module that needs it import the one lock defined here.
env_lock = threading.Lock()

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Confidence instruction injected automatically when schema has the field ───
# Placed at the end of the system prompt so it doesn't crowd the main content.
_CONFIDENCE_INSTRUCTION = """
CONFIDENCE AND CLARIFICATION:
If you have enough information to proceed, set confidence="high" or "medium" and
leave clarification_question as null.
Only set confidence="low" when the task is genuinely ambiguous in a way that would
cause the wrong output — and populate clarification_question with a single specific
question whose answer would resolve it.
Do NOT set low confidence for stylistic preferences or minor implementation details.
"""


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


def _get_output_token_cap(stage: str) -> Optional[int]:
    """
    Read the hard output-length cap for a stage from routing.yaml
    (output_token_caps.<stage>), for passing as max_tokens on the API call.
    Returns None (no cap — unbounded, previous behaviour) if the stage
    isn't listed or the config can't be read.
    """
    import yaml as _yaml
    try:
        rp = Path(__file__).parent.parent / "config" / "routing.yaml"
        with open(rp) as f:
            rcfg = _yaml.safe_load(f)
        cap = rcfg.get("output_token_caps", {}).get(stage)
        return int(cap) if cap else None
    except Exception:
        return None


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


# ── Escalation ladders ────────────────────────────────────────────────────────
# See config/models.yaml's escalation_ladders: block and
# docs/pipeline-profile-escalation-design.md §2.3/§2.4 for the full design.
# A stage escalates ONE step at a time on either of two triggers:
#   (a) TruncatedOutputError — the current model didn't finish within
#       routing.yaml's output_token_caps[stage] safety net
#   (b) TaskClassification.confidence == "low" (classify/plan stages only)
# Escalating from a role's current position tries the NEXT entry on
# escalation_ladders[role]; it never jumps straight to the ladder's end
# (that's what an "ultra" profile is for — see resolve_ultra_model below).

def get_escalation_ladder(role: str) -> list[str]:
    """Return the ordered list of model_ids to escalate `role` through."""
    cfg = _load_config()
    return cfg.get("escalation_ladders", {}).get(role, []) or []


def next_escalation_model(role: str, current_model_id: str) -> Optional[str]:
    """
    Given the model a role is CURRENTLY on, return the next model up its
    ladder, or None if there's nowhere left to escalate to (empty/exhausted
    ladder). current_model_id may be the role's default (from roles:) or
    an already-escalated model from an earlier step this run — either way,
    this only ever advances one position past whatever's passed in.
    """
    ladder = get_escalation_ladder(role)
    if not ladder:
        return None

    default_model = resolve_role(role)
    if current_model_id == default_model:
        return ladder[0]

    if current_model_id in ladder:
        idx = ladder.index(current_model_id)
        if idx + 1 < len(ladder):
            return ladder[idx + 1]
    return None


def resolve_ultra_model(role: str) -> str:
    """
    The model a role uses under the "ultra" profile: its own
    escalation_ladders[role] FINAL entry, or its normal roles: default if
    the ladder is empty (already at the ceiling, or deliberately
    unescalated like search_query/gatekeeper). This replaces the old
    _ULTRA_ROLE_MAP indirection — ultra is "start pre-escalated to where a
    fully-escalated run would end up," not a separate dedicated model.
    """
    ladder = get_escalation_ladder(role)
    return ladder[-1] if ladder else resolve_role(role)


# ── Prompt loading ─────────────────────────────────────────────────────────────

_prompt_cache: dict[str, dict] = {}

def load_prompt(role: str) -> dict:
    """
    Load and cache the YAML prompt definition for a role.

    Looks for: config/prompts/{role}.yaml
    Returns the parsed YAML dict, or an empty dict if not found.
    The dict may contain:
      system        — system prompt string (may contain {template_vars})
      user_template — user message template string (may contain {template_vars})
      thinking      — bool override (optional; overridden by call_role kwargs)
      budget_tokens — int override (optional; overridden by call_role kwargs)
    """
    if role in _prompt_cache:
        return _prompt_cache[role]

    prompt_path = Path(__file__).parent.parent / "config" / "prompts" / f"{role}.yaml"
    if not prompt_path.exists():
        log.debug("No prompt YAML found for role '%s' at %s", role, prompt_path)
        _prompt_cache[role] = {}
        return {}

    with open(prompt_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _prompt_cache[role] = data
    log.debug("Loaded prompt YAML for role '%s'", role)
    return data


def build_messages_from_prompt(
    role:             str,
    template_vars:    dict,
    response_schema:  Optional[Type[BaseModel]] = None,
    extra_messages:   Optional[list[dict]]      = None,
) -> list[dict]:
    """
    Build a messages list from a role's YAML prompt definition.

    Args:
        role:            Role name (e.g. 'plan', 'draft', 'classify')
        template_vars:   Variables to substitute into system/user templates.
                         Missing keys are left as-is (no KeyError).
        response_schema: If the schema has a 'confidence' field,
                         the confidence instruction is appended to the system prompt.
        extra_messages:  Additional messages to append after system+user
                         (used for retry loops, multi-turn, etc.)

    Returns:
        List of {role, content} dicts ready for the model.

    Raises:
        ValueError: if no YAML exists for this role and no extra_messages supplied.
    """
    prompt = load_prompt(role)

    if not prompt and not extra_messages:
        raise ValueError(
            f"No prompt YAML found for role '{role}' and no messages supplied. "
            f"Create config/prompts/{role}.yaml or pass messages directly."
        )

    messages: list[dict] = []

    # ── System prompt ──────────────────────────────────────────────────────────
    system_text = prompt.get("system", "")
    if system_text:
        # Safe format — ignore missing keys rather than raising KeyError
        system_text = _safe_format(system_text, template_vars)

        # Append confidence instruction if schema supports it
        if response_schema is not None and _schema_has_confidence(response_schema):
            system_text = system_text.rstrip() + "\n" + _CONFIDENCE_INSTRUCTION

        messages.append({"role": "system", "content": system_text})

    # ── User message ───────────────────────────────────────────────────────────
    user_template = prompt.get("user_template", "")
    if user_template:
        user_text = _safe_format(user_template, template_vars)
        messages.append({"role": "user", "content": user_text})

    # ── Extra messages (retry turns, multi-turn context) ───────────────────────
    if extra_messages:
        messages.extend(extra_messages)

    return messages


def _safe_format(template: str, vars: dict) -> str:
    """
    Format a template string with vars, leaving unresolved {keys} intact.
    Prevents KeyError when a template has optional slots.
    """
    try:
        return template.format_map(_DefaultDict(vars))
    except Exception:
        return template


class _DefaultDict(dict):
    """Returns the key wrapped in braces for missing keys — safe format_map."""
    def __missing__(self, key):
        return "{" + key + "}"


def _schema_has_confidence(schema: Type[BaseModel]) -> bool:
    """Return True if the schema has a 'confidence' field."""
    return "confidence" in schema.model_fields


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


def _extract_thinking_partial(raw_content: str) -> tuple[str, str, Optional[str]]:
    """
    Truncation-aware variant of _extract_thinking.

    _extract_thinking's regex requires a CLOSING </think> tag — on a
    response cut off mid-thought (the common case for a truncation: the
    model was still reasoning when max_tokens hit), there is no closing
    tag, so the normal regex falls through to its "no think tags" branch
    and dumps the entire unterminated raw_content — thinking and all —
    into `answer`, leaving `thinking_block` empty. That's backwards for a
    truncated call: the content between an unclosed <think> and the cutoff
    IS the thinking, not the answer, and is exactly what a person wants to
    see when asking "what was it doing with its tokens?". This handles
    that case explicitly:
      - closed <think>...</think>answer  → same split as _extract_thinking
      - unclosed <think>...(cut off)     → thinking_block = everything
        after <think>, answer = "" (there is no real answer yet)
      - no <think> tag at all            → thinking_block = "",
        answer = raw_content (matches _extract_thinking's fallback)
    """
    closed = re.search(r"<think>(.*?)</think>(.*)", raw_content, re.DOTALL | re.IGNORECASE)
    if closed:
        thinking_block = closed.group(1).strip()
        answer         = closed.group(2).strip()
    else:
        opened = _THINK_OPEN.search(raw_content)
        if opened:
            thinking_block = raw_content[opened.end():].strip()
            answer         = ""
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
    """
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


# ── Iteration-aware artifact writing ───────────────────────────────────────────

def write_iteration_artifact(run_dir: str, filename: str, content: str, iteration: int) -> str:
    """
    Write an artifact that a correction loop can produce more than once
    per run (fixed.json, verdict.json, draft.json on redraft,
    critique.json) to BOTH its flat, always-latest path (run_dir/filename
    — unchanged behaviour, still what state's *_path fields point to and
    what a fresh single-pass run without any loop iterations produces)
    AND a numbered snapshot under run_dir/iterations/<iteration>/filename.

    Without the snapshot, nodes/bugfixer.py, validator.py, and
    drafter.py each wrote straight to the flat path on every call —
    correct for a run that never loops, but for a run that goes through
    e.g. 4 rounds of bugfix -> audit -> validate before either passing or
    giving up, each round's write silently clobbered the previous one.
    GET /run (server.py's _read_artifacts_from/_ARTIFACT_FILENAMES) could
    then only ever show the LAST iteration's fixed.json/verdict.json —
    iterations 0-2 of a 4-iteration run left no trace anywhere, even
    though the loop genuinely ran and (per validate_node's verdict
    history) kept finding different issues each time. This is purely
    additive — every existing *_path field, and everything that reads
    the flat file, keeps working exactly as before; the snapshot is
    extra, not a replacement.

    iteration is whatever the calling node already has in state
    (state.get("iteration", 0)) — the SAME loop counter validate_node
    increments and route_after_validate/route_after_bugfix key their
    routing decisions on, so "iteration 2's fixed.json" here means
    exactly the fixed.json that iteration 2's validate_node evaluated,
    with no separate counter to keep in sync.

    Returns the flat path (str), same as each call site already returns
    from its own local write today — so call sites need only replace
    their `Path(x).write_text(...)` + local path variable with a call to
    this function and use of its return value, no other logic changes.
    """
    flat_path = str(Path(run_dir) / filename)
    Path(flat_path).write_text(content, encoding="utf-8")

    try:
        snapshot_dir = Path(run_dir) / "iterations" / str(iteration)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / filename).write_text(content, encoding="utf-8")
    except Exception as e:
        # Snapshot is a debugging nicety layered on top of the flat file,
        # which is the one everything else in the pipeline actually
        # depends on (state's *_path fields, redraft/re-validate logic
        # reading the in-memory object rather than the file). A disk
        # issue here should not fail the node or lose the real artifact.
        log.warning(
            "write_iteration_artifact: failed to write iteration snapshot "
            "for %s (iteration=%d) under %s: %s — flat file was still written",
            filename, iteration, run_dir, e,
        )

    return flat_path


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
    retries:    int   = 0,
    load_ms:    float = 0.0,
    ttft_ms:    float = 0.0,
    think_ratio:float = 0.0,
    memory_usage=None,   # Optional[clients.model_memory.ModelMemoryUsage]
) -> None:
    """
    Append one line to {run_dir}/stages.log and push to Langfuse.

    memory_usage is only non-None on the (rare) stage call where a fresh
    model load actually happened during this call — see call_model's
    "did_load" check and model_memory.get_load_memory's docstring for why
    it's not read/attached on every stage. Most stage log lines will have
    no memory field at all, which is correct: reporting a model's memory
    footprint again on every stage that happens to run after its load
    would be redundant at best and misleading if the number were ever
    reused stale across a later flash-swap.
    """
    log_path = Path(run_dir) / "stages.log"
    entry = {
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage":      stage,
        "model":      model_name,
        "prompt_hash":prompt_hash,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 1),
        "load_ms":    round(load_ms, 1),
        "ttft_ms":    round(ttft_ms, 1),
        "think_ratio":round(think_ratio, 2),
        "status":     status,
        "retries":    retries,
    }
    if memory_usage is not None:
        entry["memory_mib"] = {
            "gpu_total":  round(memory_usage.gpu_total_mib, 1)  if memory_usage.gpu_total_mib  is not None else None,
            "host_total": round(memory_usage.host_total_mib, 1) if memory_usage.host_total_mib is not None else None,
            "model":      memory_usage.model_mib,
            "kv_cache":   memory_usage.kv_cache_mib,
            "recurrent":  memory_usage.recurrent_mib,
            "compute":    memory_usage.compute_mib,
        }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    try:
        from langfuse.decorators import langfuse_context
        langfuse_context.update_current_observation(
            metrics={
                "model_load_ms": round(load_ms, 1),
                "ttft_ms": round(ttft_ms, 1),
                "thinking_ratio": round(think_ratio, 2)
            },
            tags=[stage, model_name]
        )
    except ImportError:
        pass


# ── Streaming completion helper ───────────────────────────────────────────────

class TruncatedOutputError(RuntimeError):
    """
    Raised when a model call hits the max_tokens cap before producing a
    complete response. Distinct from a parse failure — the content isn't
    malformed, it's just cut off — so callers can surface a clear message
    ("hit the Nnnn token limit") instead of silently propagating "".

    thinking_block / partial_answer carry whatever content the model
    produced before the cutoff (see call_model's truncation branch, which
    extracts these via _extract_thinking_partial BEFORE raising this).
    Previously this exception carried no content at all, so a truncated
    call gave no visibility into what the model had actually spent its
    tokens on — this is what lets graph.py's node-retry wrapper surface
    "here's what it was thinking about when it ran out of room" and lets
    describe_node (which builds this same content itself, see below)
    show the same thing.
    """
    def __init__(
        self, stage: str, cap: int, tokens_out: int,
        thinking_block: str = "", partial_answer: str = "",
    ):
        self.stage          = stage
        self.cap            = cap
        self.tokens_out     = tokens_out
        self.thinking_block = thinking_block
        self.partial_answer = partial_answer
        super().__init__(
            f"[{stage}] response truncated at max_tokens={cap} "
            f"({tokens_out} tokens generated, no stop condition reached)"
        )


def _stream_completion(
    client,
    model_id:   str,
    messages:   list,
    temp:       float,
    top_p:      float,
    extra_body,
    stage:      str,
    max_tokens: Optional[int] = None,
):
    chunks      = []
    token_count = 0
    in_think    = False
    think_toks  = 0
    usage       = None
    last_log    = 0
    finish_reason = None

    start_ts    = time.perf_counter()
    ttft_ms     = 0.0

    try:
        stream = client.chat.completions.create(
            model      = model_id,
            messages   = messages,
            temperature= temp,
            top_p      = top_p,
            max_tokens = max_tokens,
            extra_body = extra_body,
            stream     = True,
            # Without this, OpenAI-compatible streaming responses never
            # populate chunk.usage on any chunk — `usage` stays None for
            # the whole loop below, which is why tokens_in/tokens_out
            # were showing 0/0 in stages.log for every streamed call,
            # success or failure. This asks the server to emit one final
            # chunk with usage populated right before the stream closes.
            stream_options = {"include_usage": True},
        )

        for chunk in stream:
            if ttft_ms == 0.0:
                ttft_ms = (time.perf_counter() - start_ts) * 1000

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                text = delta.content
                chunks.append(text)
                token_count += 1

                combined = "".join(chunks)
                if "<think>" in combined and not in_think:
                    in_think = True
                if in_think and "</think>" not in combined:
                    think_toks += 1

                if token_count - last_log >= 100:
                    if in_think and "</think>" not in combined:
                        log.debug("[%s] thinking... %d tokens", stage, think_toks)
                    else:
                        log.debug("[%s] generating... %d tokens", stage, token_count)
                    last_log = token_count

            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

    except Exception as e:
        log.warning("[%s] streaming failed (%s), falling back to non-streaming", stage, e)
        resp = client.chat.completions.create(
            model=model_id, messages=messages, temperature=temp, top_p=top_p,
            max_tokens=max_tokens, extra_body=extra_body,
        )
        finish_reason = resp.choices[0].finish_reason
        return (
            resp.choices[0].message.content or "", resp.usage, 0.0, 0, 0, finish_reason
        )

    full_content = "".join(chunks)
    log.debug("[%s] completed: %d total tokens", stage, token_count)
    return full_content, usage, ttft_ms, think_toks, token_count, finish_reason


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
    output_cap_override: Optional[int] = None,
) -> T:
    """
    Make a structured model call via Instructor.

    Args:
        model_id:        Model identifier from models.yaml (e.g. '9b', '35b')
        messages:        List of {role, content} dicts (already built).
        response_schema: Pydantic model class to parse the response into.
        stage:           Pipeline stage name for logging (e.g. 'plan', 'draft').
        run_dir:         Path to the current run directory for log files.
        thinking:        Override thinking mode. None = use model default.
        budget_tokens:   Override thinking budget. None = use routing.yaml value.
        compress_system: If True, apply LLMLingua-2 to system message content.
        compress_ratio:  Compression ratio if compress_system is True.
        max_retries:     Instructor retry attempts on malformed output.
        skip_nowait:     If True, skip NoWait logit bias (e.g. chess reasoning).
        output_cap_override: If set, replaces the routing.yaml
            output_token_caps.<stage> value for just this call, without
            touching routing.yaml or affecting any other call. Used by
            the truncation-retry path (see graph.py's node-retry wrapper)
            to re-run a single node with a higher max_tokens after a
            TruncatedOutputError.

    Returns:
        Populated instance of response_schema.
    """
    cfg        = get_model_config(model_id)
    base_url   = cfg["base_url"]
    model_name = cfg["name"]
    temp       = cfg.get("temperature", 0.6)
    top_p      = cfg.get("top_p", 0.95)

    # ── Model load (VRAM swap tracking) ───────────────────────────────────────
    load_start = time.perf_counter()
    memory_usage = None
    try:
        from clients.model_manager import ensure_model_loaded
        did_load = ensure_model_loaded(model_id)
    except Exception as e:
        log.warning("model_manager.ensure_model_loaded failed: %s", e)
        did_load = False
    load_ms = (time.perf_counter() - load_start) * 1000

    if did_load:
        # Only worth reading the log for memory data when a load actually
        # just happened — see ensure_model_loaded's docstring. Best-effort:
        # get_load_memory() never raises, returns None on any failure
        # (missing -lv 4, unparseable log, etc.) — same philosophy as
        # search_web(), a missing memory reading shouldn't affect the call.
        try:
            from clients.model_memory import get_load_memory
            logs_dir = Path(__file__).parent.parent / "logs"
            memory_usage = get_load_memory(model_id, logs_dir)
        except Exception as e:
            log.warning("model_memory.get_load_memory failed: %s", e)

    # ── Thinking settings ─────────────────────────────────────────────────────
    thinking_default = cfg.get("thinking", {}).get("default_on", False)
    use_thinking     = thinking if thinking is not None else thinking_default

    if budget_tokens is not None:
        tok_budget = budget_tokens
    else:
        tok_budget = _get_thinking_budget(stage)

    # ── LLMLingua-2 compression ───────────────────────────────────────────────
    if compress_system:
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = compress_text(msg["content"], ratio=compress_ratio)
                break

    # ── Prompt hash ───────────────────────────────────────────────────────────
    prompt_str  = json.dumps(messages, sort_keys=True)
    prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()[:12]

    # ── extra_body ────────────────────────────────────────────────────────────
    extra_body: dict[str, Any] = {}
    if use_thinking:
        # budget_tokens=-1 means unlimited — omit budget_tokens from the
        # thinking config so llama.cpp imposes no per-call cap.
        if budget_tokens is not None and budget_tokens >= 0:
            extra_body["reasoning_budget"] = budget_tokens
            extra_body["thinking"] = {
                "type":          "enabled",
                "budget_tokens": budget_tokens,
            }
        else:
            # -1 or None with thinking=True → unlimited
            extra_body["thinking"] = {"type": "enabled"}
    else:
        extra_body["thinking"] = {"type": "disabled"}

    logit_bias = _build_logit_bias(cfg) if not skip_nowait else None
    if logit_bias:
        extra_body["logit_bias"] = logit_bias

    # ── Clients ───────────────────────────────────────────────────────────────
    # timeout was previously hardcoded to 1200.0, silently ignoring the
    # configured routing.yaml http.timeout_seconds (default 7200s) — long
    # stages (large budget_tokens, big models) could exceed 20 minutes
    # legitimately and get killed here regardless of config.
    raw_client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=_get_http_timeout())
    client     = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    retries_used = 0
    start_ts     = time.perf_counter()
    output_cap   = output_cap_override if output_cap_override is not None else _get_output_token_cap(stage)

    try:
        raw_content, usage, ttft_ms, think_toks, gen_toks, finish_reason = _stream_completion(
            raw_client, cfg["model_id"], messages, temp, top_p,
            extra_body if extra_body else None, stage, max_tokens=output_cap,
        )

        # ── Extract thinking BEFORE the truncation check ────────────────────
        # raw_content holds everything the model streamed, even when
        # finish_reason=="length" — the tokens it spent thinking right up
        # to the cutoff are real content, not garbage, and are exactly
        # what a person wants to see when a call gets truncated ("what was
        # it thinking about when it ran out of room?"). Previously this
        # extraction ran AFTER the truncation raise below, which meant it
        # never ran at all on the truncated path — raw_content and the
        # thinking it contained were discarded the moment finish_reason
        # was seen. Doing it here means _write_thinking_log always fires,
        # truncated or not, and the truncated case can now attach the
        # extracted (possibly partial/unterminated) thinking block to
        # TruncatedOutputError for callers to surface.
        #
        # Use the truncation-aware extractor when finish_reason=="length":
        # a truncated call very often has an OPEN <think> with no closing
        # tag (cut off mid-thought), which _extract_thinking's regex can't
        # split — it would dump the whole unterminated block into `answer`
        # and report an empty thinking_block. _extract_thinking_partial
        # treats "opened but not closed" as "everything after <think> is
        # the thinking so far". The non-truncated path keeps using the
        # original extractor — no behaviour change for successful calls.
        will_truncate = finish_reason in ("length", "max_tokens")
        if will_truncate:
            thinking_block, answer, confidence_signal = _extract_thinking_partial(raw_content)
        else:
            thinking_block, answer, confidence_signal = _extract_thinking(raw_content)

        if thinking_block:
            _write_thinking_log(run_dir, stage, thinking_block)

        # ── Detect truncation BEFORE trying to parse ────────────────────────
        # A response cut off by max_tokens (or by the server's own context
        # limit when output_cap is None/unconfigured) isn't malformed JSON —
        # it's incomplete content. Feeding it to model_validate_json/Instructor
        # either raises an opaque parse error or, worse, an empty `answer`
        # value quietly becomes "" downstream (e.g. final.json). Fail loudly
        # here instead, with the actual cap and token count, so the stage log
        # and any caller (e.g. describe_node) can surface a real message.
        if will_truncate:
            tokens_out_trunc = usage.completion_tokens if usage else gen_toks
            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            _log_stage_entry(
                run_dir=run_dir, stage=stage, model_name=model_name, prompt_hash=prompt_hash,
                tokens_in=usage.prompt_tokens if usage else 0, tokens_out=tokens_out_trunc,
                latency_ms=elapsed_ms, status="truncated", retries=0,
                load_ms=load_ms, ttft_ms=ttft_ms,
                think_ratio=(think_toks / gen_toks) if gen_toks > 0 else 0.0,
            )
            raise TruncatedOutputError(
                stage=stage,
                cap=output_cap if output_cap is not None else tokens_out_trunc,
                tokens_out=tokens_out_trunc,
                thinking_block=thinking_block,
                partial_answer=answer,
            )

        # ── Parse structured output ────────────────────────────────────────
        try:
            result: T = response_schema.model_validate_json(answer)
            retries_used = 0
        except Exception:
            field_names = list(response_schema.model_fields.keys())
            fields_hint = ", ".join(f'"{f}": <value>' for f in field_names[:6])
            result, completion = client.chat.completions.create_with_completion(
                model      = cfg["model_id"],
                messages   = messages + [
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
                max_tokens     = output_cap,
            )
            retries_used = max_retries

        # ── Attach confidence signal from thinking block ───────────────────
        # The model writes <confidence>high</confidence> inside <think>.
        # We propagate it to the schema's confidence field if present and
        # if the model didn't already populate it with a non-default value.
        if confidence_signal and _schema_has_confidence(response_schema):
            existing = getattr(result, "confidence", None)
            # Only override if model left it at the default "high"
            # (meaning it didn't express a lower confidence in the JSON body itself)
            if existing == "high" and confidence_signal.lower() in ("medium", "low"):
                try:
                    result.confidence = confidence_signal.lower()
                except Exception:
                    pass  # frozen model or validation error — skip

        # ── Metrics and logging ────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        tokens_in  = usage.prompt_tokens     if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        think_ratio = (think_toks / gen_toks) if gen_toks > 0 else 0.0

        _log_stage_entry(
            run_dir=run_dir, stage=stage, model_name=model_name, prompt_hash=prompt_hash,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=elapsed_ms,
            status="ok", retries=retries_used,
            load_ms=load_ms, ttft_ms=ttft_ms, think_ratio=think_ratio,
            memory_usage=memory_usage,
        )

        log.info(
            "[%s] %s → %s | %d+%d tok | TTFT: %.0fms | Ratio: %.2f | %.0fms",
            stage, model_name, response_schema.__name__,
            tokens_in, tokens_out, ttft_ms, think_ratio, elapsed_ms,
        )

        return result

    except TruncatedOutputError:
        # Already logged (status="truncated", with real tokens_out) at the
        # point it was raised above — don't overwrite that with a generic
        # error:TruncatedOutputError / tokens_out=0 entry here.
        log.error("[%s] %s call failed: response truncated (see stages.log)", stage, model_name)
        raise

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        _log_stage_entry(
            run_dir=run_dir, stage=stage, model_name=model_name, prompt_hash=prompt_hash,
            tokens_in=0, tokens_out=0, latency_ms=elapsed_ms,
            status=f"error:{type(exc).__name__}", retries=retries_used,
            load_ms=load_ms, ttft_ms=0.0, think_ratio=0.0,
        )
        log.error("[%s] %s call failed: %s", stage, model_name, exc)
        raise


# ── Agentic tool calling ─────────────────────────────────────────────────────
#
# Confirmed against Qwen3.5-9B via llama.cpp (--jinja) on 2026-09-10: native
# tools= / tool_calls works, returns structured (not prose) output — see
# test_tool_calling.py. Not yet independently confirmed for 35B, but same
# model family/template conventions, so expected to behave the same;
# re-run test_tool_calling.py --port 8083 if anything here misbehaves on
# that role specifically.
#
# This is a SEPARATE code path from call_model() above, not a modification
# of it — call_model() still serves every non-tool caller (classify, draft,
# critic, validate, etc.) completely unchanged. call_model_with_tools() is
# opt-in per call site via call_role(..., tools=[...]).
#
# Deliberately does NOT reuse _stream_completion(): tool-calling responses
# need the raw tool_calls array (and reasoning_content — see below), and
# _stream_completion's chunk-accumulation logic is built around extracting
# a single flat content string, not a structured tool_calls array off a
# streamed delta. Streaming tool-call deltas is a real llama.cpp/openai-sdk
# feature but adds real complexity (accumulating partial tool_call args
# across chunks) for a fairly minor UX win on server-side pipeline calls
# nobody is watching token-by-token — non-streaming is deliberately simpler
# here and was not something the original streaming path did without
# reason, so if you need to add streaming later, treat it as a distinct
# addition rather than assuming _stream_completion can just be reused.

class ToolCallRecord(BaseModel):
    """One tool call + its result, for logging/display purposes."""
    name:      str
    arguments: dict
    result:    str


def call_model_with_tools(
    model_id:        str,
    messages:        list[dict],
    tools:           list[dict],
    tool_impls:      dict,
    response_schema: Type[T],
    stage:           str,
    run_dir:         str,
    thinking:        Optional[bool] = None,
    budget_tokens:   Optional[int]  = None,
    max_tool_rounds: int            = 4,
    max_retries:     int            = 3,
    output_cap_override: Optional[int] = None,
) -> tuple[T, list[ToolCallRecord]]:
    """
    Like call_model(), but lets the model call tools mid-generation before
    producing its final structured answer.

    Loop: send messages+tools -> if the model returns tool_calls, run each
    one via tool_impls, append the assistant tool_calls message AND a
    "tool" role message per result, and go again -> once the model returns
    no tool_calls, treat message.content as the final answer and parse it
    into response_schema exactly like call_model() does.

    max_tool_rounds caps how many times the model can call a tool before
    we force it to answer (guards against a model that keeps calling
    search_web indefinitely). Hitting the cap does not raise — the last
    response is parsed as the final answer, same as if the model had
    stopped calling tools on its own; a model this deep into tool use
    usually has enough context to answer even if it would have preferred
    one more round.

    Returns (parsed_result, tool_call_history) — the history is new
    information call_model() has no equivalent of, since no tool calls
    are possible on that path. Callers that want to log/display "what was
    searched" (e.g. writing it into a run's stage log, or surfacing it in
    the UI) should persist tool_call_history themselves; this function
    does not write it to run_dir on its own, matching call_model()'s
    existing pattern of leaving artifact-writing to callers/nodes.

    NOTE on thinking capture: llama.cpp surfaces thinking-mode output on
    tool-calling responses as a separate `reasoning_content` field on the
    message, NOT as <think>...</think> tags inside `content` the way
    call_model()/_extract_thinking expect (confirmed via
    test_tool_calling.py's raw response dump). This function captures
    reasoning_content into the thinking log the same way call_model()
    captures <think> blocks, but does NOT attempt <confidence> tag
    extraction from it — that convention was designed for the
    embedded-tag format and hasn't been validated against
    reasoning_content's structure. If you need confidence signals out of
    a tool-calling call, have the model put it in the structured
    response_schema's confidence field directly rather than relying on
    tag-scraping here.
    """
    cfg        = get_model_config(model_id)
    base_url   = cfg["base_url"]
    model_name = cfg["name"]
    temp       = cfg.get("temperature", 0.6)
    top_p      = cfg.get("top_p", 0.95)

    load_start = time.perf_counter()
    memory_usage = None
    try:
        from clients.model_manager import ensure_model_loaded
        did_load = ensure_model_loaded(model_id)
    except Exception as e:
        log.warning("model_manager.ensure_model_loaded failed: %s", e)
        did_load = False
    load_ms = (time.perf_counter() - load_start) * 1000

    if did_load:
        try:
            from clients.model_memory import get_load_memory
            logs_dir = Path(__file__).parent.parent / "logs"
            memory_usage = get_load_memory(model_id, logs_dir)
        except Exception as e:
            log.warning("model_memory.get_load_memory failed: %s", e)

    thinking_default = cfg.get("thinking", {}).get("default_on", False)
    use_thinking     = thinking if thinking is not None else thinking_default
    tok_budget       = budget_tokens if budget_tokens is not None else _get_thinking_budget(stage)

    extra_body: dict[str, Any] = {}
    if use_thinking:
        if tok_budget is not None and tok_budget >= 0:
            extra_body["reasoning_budget"] = tok_budget
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": tok_budget}
        else:
            extra_body["thinking"] = {"type": "enabled"}
    else:
        extra_body["thinking"] = {"type": "disabled"}

    raw_client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=_get_http_timeout())
    output_cap = output_cap_override if output_cap_override is not None else _get_output_token_cap(stage)

    working_messages = list(messages)  # don't mutate caller's list
    tool_history: list[ToolCallRecord] = []
    start_ts = time.perf_counter()
    rounds   = 0

    while True:
        rounds += 1
        force_final = rounds > max_tool_rounds

        try:
            resp = raw_client.chat.completions.create(
                model       = cfg["model_id"],
                messages    = working_messages,
                tools       = None if force_final else tools,
                tool_choice = "none" if force_final else "auto",
                temperature = temp,
                top_p       = top_p,
                max_tokens  = output_cap,
                extra_body  = extra_body if extra_body else None,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            _log_stage_entry(
                run_dir=run_dir, stage=stage, model_name=model_name,
                prompt_hash=hashlib.sha256(json.dumps(working_messages, sort_keys=True).encode()).hexdigest()[:12],
                tokens_in=0, tokens_out=0, latency_ms=elapsed_ms,
                status=f"error:{type(exc).__name__}", retries=0,
                load_ms=load_ms, ttft_ms=0.0, think_ratio=0.0,
            )
            log.error("[%s] %s tool-call round %d failed: %s", stage, model_name, rounds, exc)
            raise

        msg = resp.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            _write_thinking_log(run_dir, stage, reasoning)

        if msg.tool_calls and not force_final:
            log.info(
                "[%s] %s round %d: model requested %d tool call(s)",
                stage, model_name, rounds, len(msg.tool_calls),
            )
            working_messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    log.warning(
                        "[%s] tool call %s had unparseable arguments (%s): %r",
                        stage, tool_name, e, tc.function.arguments,
                    )
                    tool_args = {}

                impl = tool_impls.get(tool_name)
                if impl is None:
                    result_text = f"Error: unknown tool '{tool_name}'."
                    log.warning("[%s] model called unregistered tool '%s'", stage, tool_name)
                else:
                    try:
                        result_text = impl(tool_args)
                    except Exception as e:
                        # A tool implementation failing shouldn't crash the
                        # whole node call — same best-effort philosophy as
                        # search_web() itself. Feed the error back to the
                        # model as a tool result so it can adapt (retry
                        # differently, or answer without that tool's data)
                        # rather than the pipeline blowing up here.
                        log.warning("[%s] tool '%s' raised: %s", stage, tool_name, e)
                        result_text = f"Error running tool: {e}"

                tool_history.append(ToolCallRecord(
                    name=tool_name, arguments=tool_args, result=result_text,
                ))
                working_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })
            continue  # loop back with tool results in context

        # No tool calls (or we forced a final answer) — this is the answer.
        answer = msg.content or ""
        break

    # ── Parse structured output — same pattern as call_model() ────────────
    try:
        result: T = response_schema.model_validate_json(answer)
        retries_used = 0
    except Exception:
        field_names = list(response_schema.model_fields.keys())
        fields_hint = ", ".join(f'"{f}": <value>' for f in field_names[:6])
        client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        result, _completion = client.chat.completions.create_with_completion(
            model      = cfg["model_id"],
            messages   = working_messages + [
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
            max_tokens     = output_cap,
        )
        retries_used = max_retries

    elapsed_ms  = (time.perf_counter() - start_ts) * 1000
    tokens_in   = resp.usage.prompt_tokens     if resp.usage else 0
    tokens_out  = resp.usage.completion_tokens if resp.usage else 0
    prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12]

    _log_stage_entry(
        run_dir=run_dir, stage=stage, model_name=model_name, prompt_hash=prompt_hash,
        tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=elapsed_ms,
        status="ok", retries=retries_used,
        load_ms=load_ms, ttft_ms=0.0, think_ratio=0.0,
        memory_usage=memory_usage,
    )
    log.info(
        "[%s] %s -> %s | %d+%d tok | %d tool round(s), %d call(s) | %.0fms",
        stage, model_name, response_schema.__name__,
        tokens_in, tokens_out, rounds, len(tool_history), elapsed_ms,
    )

    return result, tool_history


# ── call_role — primary interface for all node files ──────────────────────────

class EscalationNeeded(Exception):
    """
    Raised by call_role() INSTEAD OF silently escalating, when the caller
    passed require_confirmation=True (nodes do this when
    state["human_in_the_loop"] is True — see design doc §2.6, corrected:
    "ask if unsure" means ask BEFORE escalating, not only once a stage's
    ladder is exhausted).

    call_role has no access to interrupt() and shouldn't — asking a human
    is a pipeline-control concern that belongs in the node, not the HTTP
    client layer. So call_role's job stops at "here's what I would have
    escalated to, and why" — it raises this instead of calling the next
    model, and the node catches it, calls interrupt() to ask, then either:
      - re-invokes call_role with current_model_override=next_model_id to
        actually perform the escalated call, or
      - accepts the current (truncated / low-confidence) `result` as final
        if the person declines.

    Attributes:
        stage:            the stage that wants to escalate
        current_model_id: the model that just produced `trigger_result`
        next_model_id:    the model call_role would escalate to
        trigger:          "truncation" | "low_confidence"
        result:           the low-confidence result object (None for a
                          truncation trigger, since there's no parsed
                          object to return — the caller falls back to
                          exc.__cause__ / re-raising if declined)
    """
    def __init__(self, stage, current_model_id, next_model_id, trigger, result=None):
        self.stage             = stage
        self.current_model_id  = current_model_id
        self.next_model_id     = next_model_id
        self.trigger           = trigger
        self.result            = result
        super().__init__(
            f"Stage '{stage}' wants to escalate {current_model_id} → "
            f"{next_model_id} ({trigger}) — awaiting confirmation."
        )


def call_role(
    role:            str,
    messages:        Optional[list[dict]]  = None,
    response_schema: Optional[Type[T]]     = None,
    stage:           Optional[str]         = None,
    run_dir:         str                   = "",
    thinking:        Optional[bool]        = None,
    budget_tokens:   Optional[int]         = None,
    max_retries:     int                   = 3,
    template_vars:   Optional[dict]        = None,
    extra_messages:  Optional[list[dict]]  = None,
    tools:           Optional[list[dict]]  = None,
    tool_impls:      Optional[dict]        = None,
    max_tool_rounds: int                   = 4,
    profile:               Optional[str]   = None,
    current_model_override: Optional[str]  = None,
    allow_escalation:       bool           = True,
    require_confirmation:   bool           = False,
    **kwargs,
) -> T:
    """
    Primary interface for all node files. Resolves role → model, builds
    messages from YAML prompt template, and delegates to call_model() (or
    call_model_with_tools() when tools= is supplied).

    Two usage modes:

    MODE A — YAML-driven (preferred for new/migrated nodes):
        call_role(
            role="plan",
            template_vars={"task": task_text, "ideation_block": ideation_str},
            response_schema=PlanSpec,
            stage="plan",
            run_dir=run_dir,
            thinking=True,
        )
        Messages are built automatically from config/prompts/plan.yaml.
        Confidence instruction is injected if PlanSpec has a confidence field.

    MODE B — Explicit messages (legacy / multi-turn flows):
        call_role(
            role="classify",
            messages=explicit_messages,
            response_schema=TaskClassification,
            stage="classify",
            run_dir=run_dir,
        )
        Messages are used as-is. YAML system prompt is prepended if no
        system message is present in the provided list.

    AGENTIC TOOL CALLING (either mode — pass tools= and tool_impls=):
        call_role(
            role="plan",
            template_vars={"task": task_text, ...},   # no search_block needed
            response_schema=PlanSpec,
            stage="plan",
            run_dir=run_dir,
            thinking=True,
            tools=[SEARCH_TOOL_SCHEMA],
            tool_impls=TOOL_IMPLEMENTATIONS,
        )
        When tools is non-empty, call_role returns ONLY the parsed
        response_schema instance (same return shape as the non-tool path)
        — the tool_call_history that call_model_with_tools() also produces
        is available via call_role_with_tool_history() if a caller wants
        it; call_role itself discards it to keep this function's return
        type consistent for every existing call site. tool_impls defaults
        to clients.tools.TOOL_IMPLEMENTATIONS if tools is set but
        tool_impls isn't, so passing just tools=[...] with the standard
        registry works without also importing/passing TOOL_IMPLEMENTATIONS
        by hand.

    Args:
        role:            Role name from models.yaml roles section.
        messages:        Explicit message list (Mode B). If None, template_vars required.
        response_schema: Pydantic model to parse response into.
        stage:           Stage name for logging. Defaults to role if not provided.
        run_dir:         Run directory path for logs.
        thinking:        Override thinking mode. None = use YAML/model default.
        budget_tokens:   Override thinking budget. None = use YAML/routing.yaml value.
        max_retries:     Instructor retry count.
        template_vars:   Dict of variables to render into YAML templates (Mode A).
        extra_messages:  Additional turns appended after system+user (both modes).
        tools:           Tool schemas (OpenAI tools= format) to expose to the
                          model for this call. None/empty = no tool calling,
                          identical behaviour to before this parameter existed.
        tool_impls:      name -> callable(args: dict) -> str. Defaults to
                          clients.tools.TOOL_IMPLEMENTATIONS when tools is set.
        max_tool_rounds: Cap on tool-call round-trips before forcing a final
                          answer. See call_model_with_tools's docstring.
        profile:         Active pipeline_profiles name for this run (routing.yaml).
                          "ultra" resolves this role straight to
                          resolve_ultra_model(role) up front (ladder's final
                          entry), bypassing the normal roles:/escalation walk
                          below entirely — ultra runs don't discover their
                          way to the ceiling one truncation at a time.
                          Any other profile just uses its role_overrides
                          (pipeline_profiles.<profile>.role_overrides in
                          routing.yaml) in place of roles: as the STARTING
                          model, same as before. None = behave exactly like
                          the pre-profile default (use roles: as-is).
        current_model_override: If this stage has already escalated earlier
                          in the SAME run (state["escalated_models"][stage]),
                          pass that model_id here so a second escalation
                          event continues walking the ladder from where it
                          left off instead of restarting from the profile's
                          starting model. None = start from profile/roles:.
        allow_escalation: Set False to disable the escalation walk for this
                          call entirely (e.g. a role with no ladder, or a
                          caller that wants the old raise-immediately
                          behaviour). Defaults True.
        require_confirmation: When allow_escalation would otherwise fire
                          (TruncatedOutputError, or confidence=="low"),
                          raise EscalationNeeded instead of silently
                          calling the next model. Nodes set this from
                          state["human_in_the_loop"] (design doc §2.6,
                          corrected: "ask if unsure" gates escalation
                          itself, not just the final halt after the
                          ladder's exhausted) — see EscalationNeeded's
                          docstring for the full node-side flow. Ignored
                          if allow_escalation=False or there's no next
                          model on this role's ladder (nothing to confirm).
        **kwargs:        Forwarded to call_model() (e.g. skip_nowait, compress_system).
                          Not used on the tools= path (call_model_with_tools
                          doesn't accept compress_system/skip_nowait today).

    Returns:
        Populated instance of response_schema. If escalation occurred
        during this call, the returned instance carries two extra
        attributes (set via object.__setattr__ since Pydantic models are
        normally immutable-by-field-set): `_escalated_from` (the model_id
        the call started on) and `_escalated_to` (the model_id it
        succeeded on), both None if no escalation happened. Node files
        that care (currently just classifier.py/planner.py, the two
        confidence-bearing stages) read these back to update
        state["escalated_models"]/state["escalation_history"]; every
        other call site can ignore them exactly as before.
    """
    stage    = stage or role

    # ── Starting model resolution ────────────────────────────────────────────
    # Priority, highest first:
    #   1. current_model_override — this stage already escalated earlier in
    #      this run; resume from there rather than restarting at the top.
    #   2. profile == "ultra" — resolve straight to this role's escalation
    #      ladder's FINAL entry (resolve_ultra_model). Replaces the old
    #      PIPELINE_ULTRA / _ULTRA_ROLE_MAP indirection: ultra no longer
    #      remaps onto a separate dedicated model+role, it just starts this
    #      role pre-escalated to where a fully-escalated run would end up.
    #      Budget stays whatever routing.yaml's thinking_budgets[stage] says
    #      (per design doc §2.3 — budgets are keyed by stage, not model;
    #      ultra_* thinking_budgets keys, if still present in routing.yaml,
    #      are legacy and no longer read here).
    #   3. profile has a role_overrides entry for this role (short/medium/
    #      long profiles in routing.yaml's pipeline_profiles) — use it.
    #   4. models.yaml's roles: map — the original default behaviour.
    if current_model_override:
        model_id = current_model_override
    elif profile == "ultra":
        model_id = resolve_ultra_model(role)
        if model_id != resolve_role(role):
            log.debug("Ultra profile: role '%s' starts on '%s' (ladder final entry)", role, model_id)
    else:
        model_id = resolve_role(role)
        if profile:
            try:
                cfg = _load_config()
                routing_path = Path(__file__).parent.parent / "config" / "routing.yaml"
                with open(routing_path) as f:
                    routing_cfg = yaml.safe_load(f) or {}
                overrides = (
                    routing_cfg.get("pipeline_profiles", {})
                    .get(profile, {})
                    .get("role_overrides", {})
                )
                if role in overrides:
                    model_id = overrides[role]
                    log.debug("Profile '%s': role '%s' overridden to '%s'", profile, role, model_id)
            except Exception:
                log.warning("Could not read pipeline_profiles for profile '%s' — using roles: default", profile)

    # ── Custom pipeline per-step overrides ────────────────────────────────
    # Set by pipeline/custom_graph.py._wrap_existing_node when a custom
    # pipeline reuses a built-in node but wants a different model/budget
    # than that node's normal role assignment. Scoped to a single node
    # call via a context-managed env var, not a persistent config change.
    # Takes precedence over ultra remapping — an explicit per-step override
    # in a custom pipeline definition is a more specific instruction than
    # the global ultra mode toggle.
    step_model_override = os.environ.get("PIPELINE_STEP_MODEL_OVERRIDE")
    if step_model_override:
        log.debug("Custom pipeline override: role '%s' -> model '%s'", role, step_model_override)
        model_id = step_model_override
    step_budget_override = os.environ.get("PIPELINE_STEP_BUDGET_OVERRIDE")
    if step_budget_override is not None:
        budget_tokens = int(step_budget_override)

    # ── Truncation-retry output cap override ──────────────────────────────
    # Set by graph.py's node-retry wrapper (_wrap_node_for_truncation_retry)
    # for exactly one call: re-running a node that just raised
    # TruncatedOutputError, with a higher max_tokens than routing.yaml's
    # output_token_caps.<stage> would normally allow. Distinct from
    # PIPELINE_STEP_BUDGET_OVERRIDE above, which only affects the
    # *thinking* budget — the output cap (max_tokens on the completion
    # call, the thing that actually causes finish_reason="length") is a
    # separate value read via _get_output_token_cap(stage) inside
    # call_model, so bumping budget_tokens alone would not have fixed a
    # truncation caused by the output cap.
    output_cap_override = kwargs.pop("output_cap_override", None)
    step_output_cap_override = os.environ.get("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE")
    if output_cap_override is None and step_output_cap_override is not None:
        output_cap_override = int(step_output_cap_override)

    # ── Resolve thinking mode from YAML if not explicitly overridden ─────────
    # budget_tokens intentionally NOT read from YAML — routing.yaml is the
    # single source of truth for all thinking budgets. See thinking_budgets
    # section in config/routing.yaml.
    prompt_def = load_prompt(role)
    if thinking is None and "thinking" in prompt_def:
        thinking = bool(prompt_def["thinking"])

    # ── Build messages ─────────────────────────────────────────────────────────
    if messages is None:
        # Mode A: build from YAML template
        if not prompt_def:
            raise ValueError(
                f"call_role(role='{role}'): no messages supplied and no YAML prompt found. "
                f"Either pass messages= or create config/prompts/{role}.yaml."
            )
        final_messages = build_messages_from_prompt(
            role            = role,
            template_vars   = template_vars or {},
            response_schema = response_schema,
            extra_messages  = extra_messages,
        )
    else:
        # Mode B: use provided messages
        # If no system message present, prepend the YAML system prompt if available
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and prompt_def.get("system"):
            system_text = _safe_format(prompt_def["system"], template_vars or {})
            if response_schema and _schema_has_confidence(response_schema):
                system_text = system_text.rstrip() + "\n" + _CONFIDENCE_INSTRUCTION
            messages = [{"role": "system", "content": system_text}] + messages

        if extra_messages:
            messages = messages + extra_messages

        final_messages = messages

    # ── Escalation-aware dispatch ─────────────────────────────────────────────
    # See docs/pipeline-profile-escalation-design.md §2.3, and §2.6 as
    # corrected: "ask if unsure" (human_in_the_loop) gates escalation
    # itself — the node asks BEFORE calling a bigger model, not only once
    # a stage's ladder is already exhausted. require_confirmation is how a
    # node expresses that: instead of this function silently calling
    # _dispatch(next_model), it raises EscalationNeeded and lets the node
    # decide (via interrupt()) whether to proceed. When
    # require_confirmation is False (set-and-forget), behaviour is
    # unchanged from before — escalate immediately, one step, same as
    # always. Either way, a still-failing result after ONE escalation
    # attempt (or after a confirmed one) propagates normally, so
    # graph.py's _wrap_node_for_truncation_retry (the ladder-exhausted /
    # manual-retry-with-higher-cap fallback, kept per design doc §2.5/§4
    # item 4) still catches genuinely ladder-exhausted truncations.
    starting_model_id = model_id
    escalated_to: Optional[str] = None

    def _dispatch(mid: str):
        if tools:
            impls = tool_impls
            if impls is None:
                from clients.tools import TOOL_IMPLEMENTATIONS as _default_tool_impls
                impls = _default_tool_impls
            result, _tool_history = call_model_with_tools(
                model_id        = mid,
                messages        = final_messages,
                tools           = tools,
                tool_impls      = impls,
                response_schema = response_schema,
                stage           = stage,
                run_dir         = run_dir,
                thinking        = thinking,
                budget_tokens   = budget_tokens,
                max_tool_rounds = max_tool_rounds,
                max_retries     = max_retries,
                output_cap_override = output_cap_override,
            )
            return result
        return call_model(
            model_id        = mid,
            messages        = final_messages,
            response_schema = response_schema,
            stage           = stage,
            run_dir         = run_dir,
            thinking        = thinking,
            budget_tokens   = budget_tokens,
            max_retries     = max_retries,
            output_cap_override = output_cap_override,
            **kwargs,
        )

    try:
        result = _dispatch(model_id)
    except TruncatedOutputError:
        next_model = next_escalation_model(role, model_id) if allow_escalation else None
        if not next_model:
            raise
        if require_confirmation:
            log.info(
                "Stage '%s' truncated on '%s' — escalation to '%s' needs confirmation",
                stage, model_id, next_model,
            )
            raise EscalationNeeded(stage, model_id, next_model, "truncation") from None
        log.warning(
            "Stage '%s' truncated on '%s' — escalating to '%s' (ladder step)",
            stage, model_id, next_model,
        )
        result = _dispatch(next_model)
        escalated_to = next_model

    # Also escalate once on low confidence for schemas that carry it
    # (classify/plan — TaskClassification/PlanSpec both have a confidence
    # field). Only fires if the FIRST attempt didn't already escalate for
    # truncation above (one escalation event per call_role invocation,
    # per design doc §2.3's "one step per failure").
    if (
        escalated_to is None
        and allow_escalation
        and getattr(result, "confidence", None) == "low"
    ):
        next_model = next_escalation_model(role, model_id)
        if next_model:
            if require_confirmation:
                log.info(
                    "Stage '%s' confidence=low on '%s' — escalation to '%s' needs confirmation",
                    stage, model_id, next_model,
                )
                raise EscalationNeeded(stage, model_id, next_model, "low_confidence", result=result)
            log.warning(
                "Stage '%s' confidence=low on '%s' — escalating to '%s' (ladder step)",
                stage, model_id, next_model,
            )
            try:
                retried = _dispatch(next_model)
                result = retried
                escalated_to = next_model
            except TruncatedOutputError:
                # The escalated model truncated instead of improving —
                # let it propagate; graph.py's wrapper / human-in-the-loop
                # fallback (or set-and-forget best-effort) takes it from here.
                raise

    if escalated_to:
        try:
            object.__setattr__(result, "_escalated_from", starting_model_id)
            object.__setattr__(result, "_escalated_to", escalated_to)
        except Exception:
            pass
    else:
        try:
            object.__setattr__(result, "_escalated_from", None)
            object.__setattr__(result, "_escalated_to", None)
        except Exception:
            pass

    return result


def call_role_with_tool_history(*args, **kwargs) -> tuple[T, list["ToolCallRecord"]]:
    """
    Same as call_role(role, ..., tools=[...]), but also returns the
    ToolCallRecord history (which tools were called, with what arguments,
    and what came back) instead of discarding it.

    Only meaningful when tools= is passed — raises ValueError otherwise,
    since without tools there's no history to return and callers should
    just use call_role() directly.

    Exists for callers that want to persist/display what was searched
    (e.g. writing a "search history" artifact to run_dir, or surfacing
    "🔍 searched for: ..." somewhere) without every ordinary call_role()
    caller having to change its unpacking (result vs. (result, history))
    just because tool calling exists elsewhere in the codebase.
    """
    tools = kwargs.get("tools")
    if not tools:
        raise ValueError(
            "call_role_with_tool_history() requires tools=[...] — "
            "use call_role() directly for non-tool calls."
        )

    role  = args[0] if args else kwargs.pop("role")
    rest_args = args[1:] if args else ()

    model_id = resolve_role(role)
    stage    = kwargs.get("stage") or role
    tool_impls = kwargs.get("tool_impls")
    if tool_impls is None:
        from clients.tools import TOOL_IMPLEMENTATIONS as _default_tool_impls
        tool_impls = _default_tool_impls

    prompt_def = load_prompt(role)
    thinking = kwargs.get("thinking")
    if thinking is None and "thinking" in prompt_def:
        thinking = bool(prompt_def["thinking"])

    messages = kwargs.get("messages")
    if messages is None:
        final_messages = build_messages_from_prompt(
            role            = role,
            template_vars   = kwargs.get("template_vars") or {},
            response_schema = kwargs.get("response_schema"),
            extra_messages  = kwargs.get("extra_messages"),
        )
    else:
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and prompt_def.get("system"):
            system_text = _safe_format(prompt_def["system"], kwargs.get("template_vars") or {})
            response_schema = kwargs.get("response_schema")
            if response_schema and _schema_has_confidence(response_schema):
                system_text = system_text.rstrip() + "\n" + _CONFIDENCE_INSTRUCTION
            messages = [{"role": "system", "content": system_text}] + messages
        extra_messages = kwargs.get("extra_messages")
        if extra_messages:
            messages = messages + extra_messages
        final_messages = messages

    return call_model_with_tools(
        model_id        = model_id,
        messages        = final_messages,
        tools           = tools,
        tool_impls      = tool_impls,
        response_schema = kwargs.get("response_schema"),
        stage           = stage,
        run_dir         = kwargs.get("run_dir", ""),
        thinking        = thinking,
        budget_tokens   = kwargs.get("budget_tokens"),
        max_tool_rounds = kwargs.get("max_tool_rounds", 4),
        max_retries     = kwargs.get("max_retries", 3),
        output_cap_override = kwargs.get("output_cap_override"),
    )