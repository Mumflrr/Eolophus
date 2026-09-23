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

# ── Cooperative run cancellation ─────────────────────────────────────────────
# DELETE /run/{uuid} used to only write status="cancelled" to run.json. Nothing
# ever told the worker thread to stop, so app_graph.invoke() kept executing
# every remaining node — and since the executor is single-worker, any new run
# or chat turn queued behind it (while its run.json already said "running").
# This is the "tell it" half: the API layer calls request_cancel(); the worker
# polls at each model-call boundary and inside the streaming loop, and raises
# RunCancelled, which unwinds through LangGraph to the thread function.
#
# RunCancelled derives from BaseException, NOT Exception, on purpose: this
# codebase has many `except Exception` blocks — including _stream_completion's
# non-streaming fallback, which would re-issue the whole request — that would
# otherwise swallow it.
#
# _active_run is process-global. That is only correct because
# state.executor is ThreadPoolExecutor(max_workers=1). If that ever changes,
# replace it with a ContextVar (and verify LangGraph copies context into its
# node threads before relying on that).
class RunCancelled(BaseException):
    """Raised inside a pipeline worker thread once its run has been cancelled."""


_cancelled_runs: set[str] = set()
_active_run: Optional[str] = None


def request_cancel(run_uuid: str) -> None:
    """Called from the API layer (DELETE /run/{uuid}) — any thread."""
    _cancelled_runs.add(run_uuid)


def clear_cancel(run_uuid: str) -> None:
    """Forget a prior cancel. Call when SUBMITTING new work for a run (e.g. a
    chat turn on a previously-cancelled run), not when the worker starts — a
    cancel that lands while the work is still queued must survive until begin_run."""
    _cancelled_runs.discard(run_uuid)


def begin_run(run_uuid: str) -> None:
    """Called first thing in a worker thread. Aborts immediately if the run was
    cancelled while it was still waiting in the executor queue."""
    global _active_run
    _active_run = run_uuid
    check_cancelled()


def end_run() -> None:
    global _active_run
    _active_run = None


def is_cancelled() -> bool:
    return _active_run is not None and _active_run in _cancelled_runs


def check_cancelled() -> None:
    if is_cancelled():
        raise RunCancelled(_active_run)

# Per-step overrides for custom pipelines (pipeline/custom_graph.py) and
# the truncation-retry wrapper (pipeline/graph.py), which each need to run
# ONE node call with a different model/thinking-budget/output-cap than its
# normal role assignment, without affecting any other call. Previously
# these were process environment variables (PIPELINE_STEP_MODEL_OVERRIDE,
# PIPELINE_STEP_BUDGET_OVERRIDE, PIPELINE_STEP_OUTPUT_CAP_OVERRIDE) mutated
# under env_lock and restored in a try/finally at every call site — correct
# only because today's setup is single-threaded per run (one GPU, ThreadPool
# Executor(max_workers=1)); both call sites' own comments already flagged
# this as a race condition waiting to happen under any future concurrency.
# ContextVars are correct under concurrency with no lock and no manual
# restore: each sets its value for the current execution context only, and
# a plain function call (not a new thread/task) automatically inherits and
# then restores the caller's context on return.
#
# env_lock itself is UNCHANGED and still guards api/server.py's separate
# PIPELINE_ULTRA/PIPELINE_FORCE_SHORT/PIPELINE_NO_ENSEMBLE env vars, which
# this module doesn't read — only the three step-level overrides moved.
import contextvars
from contextlib import contextmanager

_step_model_override      : contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("step_model_override", default=None)
_step_budget_override     : contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("step_budget_override", default=None)
_step_output_cap_override : contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("step_output_cap_override", default=None)


@contextmanager
def step_overrides(model: Optional[str] = None, budget: Optional[int] = None, output_cap: Optional[int] = None):
    """
    Scope a per-step model/thinking-budget/output-cap override to the
    wrapped block. Replaces the PIPELINE_STEP_*_OVERRIDE env vars — see
    the comment above. Only the overrides passed (non-None) are set;
    unset ones fall through to call_role's normal role/profile resolution
    exactly as before.
    """
    tokens = []
    if model is not None:
        tokens.append((_step_model_override, _step_model_override.set(model)))
    if budget is not None:
        tokens.append((_step_budget_override, _step_budget_override.set(budget)))
    if output_cap is not None:
        tokens.append((_step_output_cap_override, _step_output_cap_override.set(output_cap)))
    try:
        yield
    finally:
        for var, token in tokens:
            var.reset(token)

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
# All YAML config is read through config/loader.py — one cache for
# models.yaml and one for routing.yaml, instead of every reader (this
# module, model_manager.py, routers.py) reopening and reparsing its own
# copy. The aliases below keep every existing call site in this file (and
# describe.py, which imports these by name from clients.llm) unchanged.

from config.loader import (
    get_models_config,
    get_routing_config,
    get_thinking_budget as _get_thinking_budget,
    get_thinking_control_flag as _thinking_control_flag,
    get_http_timeout as _get_http_timeout,
    get_output_token_cap as _get_output_token_cap,
)


def _build_thinking_extra_body(use_thinking: bool, tok_budget: Optional[int]) -> dict:
    """
    The thinking part of a chat-completions request, in ONE place.

    call_model, call_model_with_tools and describe_node each used to carry their
    own copy of this logic, and the copies drifted (call_model tested the raw
    budget_tokens argument instead of the resolved tok_budget, so routing.yaml's
    thinking_budgets were silently ignored on that path). Everything that
    builds a thinking request should go through here.

      use_thinking False        -> thinking disabled
      tok_budget == 0           -> thinking disabled   (routing.yaml: 0 = "NO THINKING";
                                                        stating it as "disabled" is right
                                                        whichever key the server honours)
      tok_budget > 0            -> enabled, capped: reasoning_budget (the field
                                   llama.cpp reads) + thinking.budget_tokens
      tok_budget None or < 0    -> enabled, UNLIMITED (-1 is the explicit spelling,
                                   e.g. the ultra_* keys)
    """
    thinking_on = bool(use_thinking) and tok_budget != 0
    if not thinking_on:
        body: dict = {"thinking": {"type": "disabled"}}
    elif tok_budget is None or tok_budget < 0:
        body = {"thinking": {"type": "enabled"}}
    else:
        body = {
            "reasoning_budget": tok_budget,
            "thinking": {"type": "enabled", "budget_tokens": tok_budget},
        }
    # The `thinking` object above is an Anthropic-style field; llama.cpp does not
    # document it. The switch llama.cpp DOES document for Qwen-style templates
    # (with --jinja) is chat_template_kwargs.enable_thinking. Without it, a
    # "non-thinking" stage can still think — invisibly, in reasoning_content —
    # and a long enough prompt (see classify) turns that into a runaway that only
    # max_tokens stops. Harmless where the template ignores it. Kill switch:
    # routing.yaml  thinking_control.chat_template_kwargs: false
    if _thinking_control_flag("chat_template_kwargs", True):
        body["chat_template_kwargs"] = {"enable_thinking": thinking_on}
    return body


def get_model_config(model_id: str) -> dict:
    """Return the config block for a model_id (e.g. '9b', '35b')."""
    cfg = get_models_config()
    if model_id not in cfg["models"]:
        raise ValueError(f"Unknown model_id '{model_id}'. Check config/models.yaml.")
    return cfg["models"][model_id]


def resolve_role(role: str) -> str:
    """Resolve a role name to a model_id via config/models.yaml roles mapping."""
    cfg = get_models_config()
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
    return get_models_config().get("escalation_ladders", {}).get(role, []) or []


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

_lingua_compressor: Any = None
_lingua_unavailable: bool = False

def _get_compressor() -> Any:
    """Returns the LLMLingua PromptCompressor, or None if llmlingua isn't installed."""
    global _lingua_compressor, _lingua_unavailable
    if _lingua_unavailable:
        return None
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
            _lingua_unavailable = True
            return None
    return _lingua_compressor


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
            [text],
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
        from langfuse.decorators import langfuse_context  # pyright: ignore[reportMissingImports]
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
    reasoning_sink: Optional[list] = None,
    presence_penalty: Optional[float] = None,
):
    """
    Stream one completion. Returns (content, usage, ttft_ms, think_toks, token_count, finish_reason).

    llama.cpp (--jinja) delivers a thinking model's thinking in a separate
    `reasoning_content` delta rather than as <think> text in `content`. That
    used to be dropped on the floor here: never counted (think_ratio read 0.0 on
    every stage), never logged, and never attached to a TruncatedOutputError —
    so a stage that burned its whole cap thinking looked like it had produced
    nothing. If `reasoning_sink` is given, reasoning text is appended to it
    (the return tuple is unchanged so other callers are unaffected), and those
    tokens now count toward think_toks / token_count.
    """
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
        create_kwargs = dict(
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
        if presence_penalty is not None:
            create_kwargs["presence_penalty"] = presence_penalty
        stream = client.chat.completions.create(**create_kwargs)

        for chunk in stream:
            if is_cancelled():
                # Closing the response drops the HTTP connection, which makes
                # llama-server abort generation (frees the GPU immediately).
                # RunCancelled is a BaseException, so the `except Exception`
                # non-streaming fallback below can't catch it.
                try:
                    stream.close()
                except Exception:
                    pass
                raise RunCancelled(_active_run)
            if ttft_ms == 0.0:
                ttft_ms = (time.perf_counter() - start_ts) * 1000

            delta = chunk.choices[0].delta if chunk.choices else None
            reasoning_piece = getattr(delta, "reasoning_content", None) if delta else None
            if reasoning_piece:
                token_count += 1
                think_toks  += 1
                if reasoning_sink is not None:
                    reasoning_sink.append(reasoning_piece)
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
        fallback_reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        if fallback_reasoning and reasoning_sink is not None:
            reasoning_sink.append(fallback_reasoning)
        return (
            resp.choices[0].message.content or "", resp.usage, 0.0, 0, 0, finish_reason
        )

    full_content = "".join(chunks)
    log.debug("[%s] completed: %d total tokens", stage, token_count)
    return full_content, usage, ttft_ms, think_toks, token_count, finish_reason


# ── Core call function ────────────────────────────────────────────────────────

# ── Shared per-call setup ────────────────────────────────────────────────────
# call_model() and call_model_with_tools() both need: resolve the model's
# config, swap VRAM if this model isn't the one currently loaded (and read
# back its memory footprint if a swap just happened), resolve thinking
# mode/budget via _build_thinking_extra_body(), and build the OpenAI
# client. Used to be ~25 duplicated lines in each function.

class _PreparedCall:
    """Bag of values _prepare_call() resolves once per model call."""
    __slots__ = (
        "cfg", "model_name", "temp", "top_p", "presence_penalty",
        "load_ms", "memory_usage", "extra_body", "raw_client", "output_cap",
    )
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _prepare_call(
    model_id:             str,
    stage:                str,
    thinking:              Optional[bool],
    budget_tokens:          Optional[int],
    output_cap_override:    Optional[int],
) -> _PreparedCall:
    check_cancelled()   # don't swap models for a run that's already been cancelled
    cfg        = get_model_config(model_id)
    base_url   = cfg["base_url"]
    temp       = cfg.get("temperature", 0.6)
    top_p      = cfg.get("top_p", 0.95)
    # Only forward presence_penalty when a model config actually sets one
    # (e.g. 35b's presence_penalty: 1.0, "prevents infinite loops in <think>
    # tags") — previously declared in models.yaml but never read by any
    # call site, so it did nothing for any model, including 35b. None here
    # means omit it from the request entirely, not send an explicit 0.0.
    presence_penalty = cfg.get("presence_penalty")

    load_start = time.perf_counter()
    memory_usage = None
    try:
        from clients.model_manager import ensure_model_loaded
        did_load = ensure_model_loaded(model_id)
    except Exception as e:
        log.warning("model_manager.ensure_model_loaded failed: %s", e)
        did_load = False
    load_ms = (time.perf_counter() - load_start) * 1000
    check_cancelled()   # cancel may have landed during the swap — skip sending the prompt

    if did_load:
        # Only worth reading the log for memory data when a load actually
        # just happened. Best-effort: get_load_memory() never raises.
        try:
            from clients.model_memory import get_load_memory
            logs_dir = Path(__file__).parent.parent / "logs"
            memory_usage = get_load_memory(model_id, logs_dir)
        except Exception as e:
            log.warning("model_memory.get_load_memory failed: %s", e)

    thinking_default = cfg.get("thinking", {}).get("default_on", False)
    use_thinking      = thinking if thinking is not None else thinking_default
    tok_budget        = budget_tokens if budget_tokens is not None else _get_thinking_budget(stage)
    extra_body        = _build_thinking_extra_body(use_thinking, tok_budget)

    raw_client = OpenAI(base_url=base_url, api_key="local", max_retries=0, timeout=_get_http_timeout())
    output_cap = output_cap_override if output_cap_override is not None else _get_output_token_cap(stage)

    return _PreparedCall(
        cfg=cfg, model_name=cfg["name"], temp=temp, top_p=top_p, presence_penalty=presence_penalty,
        load_ms=load_ms, memory_usage=memory_usage,
        extra_body=extra_body, raw_client=raw_client, output_cap=output_cap,
    )


def _repair_malformed_response(
    client, model_id: str, messages: list[dict], answer: str,
    response_schema: Type[T], max_retries: int, temp: float, output_cap: Optional[int],
    presence_penalty: Optional[float] = None,
) -> T:
    """
    One retry with a corrective follow-up message, used by both call_model()
    and call_model_with_tools() when the model's answer didn't parse as
    response_schema. Small/quantized models occasionally emit a JSON
    *schema* (properties/$defs/type keys) instead of an instance, or drop a
    comma — this names the mistake explicitly rather than just resending
    the same prompt.
    """
    field_names = list(response_schema.model_fields.keys())
    fields_hint = ", ".join(f'"{f}": <value>' for f in field_names[:6])
    repair_kwargs = dict(
        model      = model_id,
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
    if presence_penalty is not None:
        repair_kwargs["presence_penalty"] = presence_penalty
    result, _completion = client.chat.completions.create_with_completion(**repair_kwargs)
    return result


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
    Make a structured model call via Instructor. See _prepare_call() for
    model/thinking/client setup shared with call_model_with_tools().

    output_cap_override: replaces routing.yaml's output_token_caps.<stage>
    for just this call — used by the truncation-retry path (graph.py's
    node-retry wrapper) to re-run one node with a higher max_tokens after
    a TruncatedOutputError, without touching routing.yaml itself.
    """
    if compress_system:
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = compress_text(msg["content"], ratio=compress_ratio)
                break

    prompt_str  = json.dumps(messages, sort_keys=True)
    prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()[:12]

    pc = _prepare_call(model_id, stage, thinking, budget_tokens, output_cap_override)
    cfg, model_name, temp, top_p = pc.cfg, pc.model_name, pc.temp, pc.top_p
    presence_penalty = pc.presence_penalty
    load_ms, memory_usage        = pc.load_ms, pc.memory_usage
    extra_body, raw_client, output_cap = pc.extra_body, pc.raw_client, pc.output_cap

    if not skip_nowait:
        logit_bias = _build_logit_bias(cfg)
        if logit_bias:
            extra_body["logit_bias"] = logit_bias

    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    retries_used = 0
    start_ts     = time.perf_counter()

    reasoning_sink: list = []
    try:
        raw_content, usage, ttft_ms, think_toks, gen_toks, finish_reason = _stream_completion(
            raw_client, cfg["model_id"], messages, temp, top_p,
            extra_body if extra_body else None, stage, max_tokens=output_cap,
            reasoning_sink=reasoning_sink, presence_penalty=presence_penalty,
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

        # Thinking that arrived in reasoning_content (see _stream_completion) is
        # merged in for LOGGING and for the truncation error only — deliberately
        # NOT for confidence_signal, which is still read from <think> text in
        # `content` exactly as before, so escalation behaviour is unchanged.
        reasoning_text = "".join(reasoning_sink)
        if reasoning_text:
            thinking_block = "\n".join(x for x in (reasoning_text, thinking_block) if x)

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
            result = _repair_malformed_response(
                client, cfg["model_id"], messages, answer, response_schema, max_retries, temp, output_cap,
                presence_penalty,
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
        # Best-effort: usage/completion may not exist if the failure was in
        # _stream_completion itself (before any usage object existed) — but
        # if it was the create_with_completion repair call that failed,
        # Instructor sometimes exposes the last raw completion via the
        # exception itself (check InstructorRetryException's attributes —
        # e.g. exc.last_completion or exc.n_attempts, depending on the
        # instructor version). Log real values when available instead of a
        # hardcoded 0/0, which reads as "the model produced nothing" when
        # what actually happened is "the model produced something that
        # didn't validate."
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
    producing its final structured answer. See _prepare_call() for setup
    shared with call_model().

    Loop: send messages+tools -> run any tool_calls via tool_impls, append
    the assistant + per-tool "tool" messages, go again -> once the model
    returns no tool_calls, parse message.content as response_schema like
    call_model() does. Raises TruncatedOutputError if any round hits the
    output cap. max_tool_rounds caps tool round-trips before the model is
    forced to answer (tool_choice="none"); hitting it does not raise.

    Returns (parsed_result, tool_call_history) for callers that want to
    log/display what was searched.

    llama.cpp puts tool-calling thinking output in a separate
    reasoning_content field, not <think> tags in content — captured into
    the thinking log the same way, but NOT run through <confidence> tag
    extraction (untested against reasoning_content's structure — put
    confidence in the response_schema directly instead).
    """
    pc = _prepare_call(model_id, stage, thinking, budget_tokens, output_cap_override)
    cfg, model_name, temp, top_p        = pc.cfg, pc.model_name, pc.temp, pc.top_p
    load_ms, memory_usage               = pc.load_ms, pc.memory_usage
    extra_body, raw_client, output_cap  = pc.extra_body, pc.raw_client, pc.output_cap
    presence_penalty = pc.presence_penalty

    working_messages = list(messages)  # don't mutate caller's list
    tool_history: list[ToolCallRecord] = []
    start_ts = time.perf_counter()
    rounds   = 0

    while True:
        check_cancelled()
        rounds += 1
        force_final = rounds > max_tool_rounds

        try:
            create_kwargs = dict(
                model       = cfg["model_id"],
                messages    = working_messages,
                tools       = None if force_final else tools,
                tool_choice = "none" if force_final else "auto",
                temperature = temp,
                top_p       = top_p,
                max_tokens  = output_cap,
                extra_body  = extra_body if extra_body else None,
            )
            if presence_penalty is not None:
                create_kwargs["presence_penalty"] = presence_penalty
            resp = raw_client.chat.completions.create(**create_kwargs)
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

        # This request is non-streaming, so a cancel that landed during it can only be
        # noticed now — stop before executing any tool calls or issuing another round.
        check_cancelled()
        msg = resp.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            _write_thinking_log(run_dir, stage, reasoning)

        # ── Truncation check — call_model has always had this; this path didn't ──
        # A round cut off by max_tokens is incomplete content, not malformed
        # JSON. Without this, a truncated final answer went straight to
        # model_validate_json, failed, and triggered the Instructor repair call
        # (up to max_retries more generations of up to output_cap tokens each)
        # on a half-written answer; a truncated TOOL round is worse, since its
        # arguments can't be trusted. Raising TruncatedOutputError here — same
        # exception, same content attached — means call_role's escalation walk /
        # EscalationNeeded confirmation and graph.py's truncation-retry wrapper
        # now cover tool-calling stages too.
        finish_reason = resp.choices[0].finish_reason
        if finish_reason in ("length", "max_tokens"):
            thinking_block, partial_answer, _ = _extract_thinking_partial(msg.content or "")
            thinking_block   = reasoning or thinking_block
            tokens_out_trunc = resp.usage.completion_tokens if resp.usage else 0
            elapsed_ms       = (time.perf_counter() - start_ts) * 1000
            _log_stage_entry(
                run_dir=run_dir, stage=stage, model_name=model_name,
                prompt_hash=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12],
                tokens_in=resp.usage.prompt_tokens if resp.usage else 0, tokens_out=tokens_out_trunc,
                latency_ms=elapsed_ms, status="truncated", retries=0,
                load_ms=load_ms, ttft_ms=0.0,
                think_ratio=len(thinking_block) / max(len(thinking_block) + len(partial_answer), 1),
                memory_usage=memory_usage,
            )
            log.error(
                "[%s] %s tool-call round %d truncated at max_tokens=%s (%d tokens; %d chars thinking, %d chars answer)",
                stage, model_name, rounds, output_cap, tokens_out_trunc, len(thinking_block), len(partial_answer),
            )
            raise TruncatedOutputError(
                stage=stage,
                cap=output_cap if output_cap is not None else tokens_out_trunc,
                tokens_out=tokens_out_trunc,
                thinking_block=thinking_block,
                partial_answer=partial_answer,
            )

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
        client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        result = _repair_malformed_response(
            client, cfg["model_id"], working_messages, answer, response_schema, max_retries, temp, output_cap,
            presence_penalty,
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
    def __init__(
        self,
        stage:            str,
        current_model_id: str,
        next_model_id:    str,
        trigger:          str,
        result:           Optional[Any] = None,
    ):
        self.stage             = stage
        self.current_model_id  = current_model_id
        self.next_model_id     = next_model_id
        self.trigger           = trigger
        # Any (not BaseModel): callers immediately use it as their own
        # response_schema type (TaskClassification, PlanSpec, ...), and this
        # exception is deliberately schema-agnostic.
        self.result: Optional[Any] = result
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
    tool_history_sink: Optional[list]      = None,
    profile:               Optional[str]   = None,
    current_model_override: Optional[str]  = None,
    allow_escalation:       bool           = True,
    require_confirmation:   bool           = False,
    **kwargs,
) -> T:
    """
    Primary interface for all node files. Resolves role → model, builds
    messages from a YAML prompt template (or uses explicit messages= if
    given), and delegates to call_model() — or call_model_with_tools()
    when tools= is supplied.

    Mode A (preferred): pass template_vars={...} and role="plan" etc. —
    messages are built from config/prompts/<role>.yaml, with a confidence
    instruction auto-injected if response_schema has a confidence field.
    Mode B (legacy/multi-turn): pass messages= explicitly; the YAML system
    prompt is prepended only if messages has no system message already.
    Passing tools=[...] works in either mode; call_role always returns
    just the parsed response_schema instance (not the tool-call history —
    use tool_history_sink=[] to also get that back, or
    call_role_with_tool_history() for the raw tuple).

    Escalation: when allow_escalation (default True) and this role has an
    escalation_ladders entry (models.yaml), a TruncatedOutputError or
    confidence=="low" result walks to the ladder's next model instead of
    failing outright. require_confirmation flips that to raising
    EscalationNeeded instead of silently retrying — nodes set this from
    state["human_in_the_loop"] — see EscalationNeeded's docstring.
    current_model_override resumes a ladder walk that already escalated
    earlier in this run, rather than restarting from the top.
    profile="ultra" skips the walk and starts straight at the ladder's
    final entry; any other profile just changes the STARTING model via
    routing.yaml's pipeline_profiles.<profile>.role_overrides.

    Returns: a response_schema instance. If escalation happened, it also
    carries `_escalated_from`/`_escalated_to` model_id attributes (set via
    object.__setattr__; both None otherwise) — node files read these back
    to update state["escalated_models"]/state["escalation_history"].

    **kwargs are forwarded to call_model() (e.g. skip_nowait,
    compress_system) — unused on the tools= path.
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
            overrides = (
                get_routing_config().get("pipeline_profiles", {})
                .get(profile, {})
                .get("role_overrides", {})
            )
            if role in overrides:
                model_id = overrides[role]
                log.debug("Profile '%s': role '%s' overridden to '%s'", profile, role, model_id)

    # ── Custom pipeline per-step overrides ────────────────────────────────
    # Set by pipeline/custom_graph.py._wrap_existing_node (via step_overrides()
    # above) when a custom pipeline reuses a built-in node but wants a
    # different model/budget than that node's normal role assignment.
    # Scoped to a single node call, not a persistent config change. Takes
    # precedence over ultra remapping — an explicit per-step override in a
    # custom pipeline definition is a more specific instruction than the
    # global ultra mode toggle.
    step_model_override = _step_model_override.get()
    if step_model_override:
        log.debug("Custom pipeline override: role '%s' -> model '%s'", role, step_model_override)
        model_id = step_model_override
    step_budget_override = _step_budget_override.get()
    if step_budget_override is not None:
        budget_tokens = step_budget_override

    # ── Truncation-retry output cap override ──────────────────────────────
    # Set by graph.py's node-retry wrapper (_wrap_node_for_truncation_retry,
    # via step_overrides() above) for exactly one call: re-running a node
    # that just raised TruncatedOutputError, with a higher max_tokens than
    # routing.yaml's output_token_caps.<stage> would normally allow.
    # Distinct from the budget override above, which only affects the
    # *thinking* budget — the output cap (max_tokens on the completion
    # call, the thing that actually causes finish_reason="length") is a
    # separate value read via _get_output_token_cap(stage) inside
    # call_model, so bumping the thinking budget alone would not have
    # fixed a truncation caused by the output cap.
    output_cap_override = kwargs.pop("output_cap_override", None)
    step_output_cap_override = _step_output_cap_override.get()
    if output_cap_override is None and step_output_cap_override is not None:
        output_cap_override = step_output_cap_override

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

    if response_schema is None:
        # call_model()/call_model_with_tools() both require a schema (Instructor
        # can't parse without one). Fail loudly and early instead of deep inside.
        raise ValueError(f"call_role('{role}', stage='{stage}') requires response_schema=")
    _schema: Type[T] = response_schema

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
                response_schema = _schema,
                stage           = stage,
                run_dir         = run_dir,
                thinking        = thinking,
                budget_tokens   = budget_tokens,
                max_tool_rounds = max_tool_rounds,
                max_retries     = max_retries,
                output_cap_override = output_cap_override,
            )
            if tool_history_sink is not None:
                tool_history_sink[:] = _tool_history
            return result
        return call_model(
            model_id        = mid,
            messages        = final_messages,
            response_schema = _schema,
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
    #
    # EXCEPTION: if the model is asking a human to resolve the ambiguity
    # (require_confirmation=True, i.e. human_in_the_loop) AND it already
    # produced a specific clarification_question, escalating first would
    # ask the WRONG question — "should we try a bigger model?" — while
    # hiding the actual question the classifier/planner wants answered.
    # A bigger model doesn't fix genuine task ambiguity (as opposed to
    # this model's own uncertainty about phrasing it couldn't otherwise
    # resolve), and the person would only see the real question at all if
    # they happened to decline escalation first (classifier.py/planner.py's
    # `esc.result is not None` fallback) — two prompts for one decision,
    # with the more useful one hidden behind the less useful one. Skip
    # straight to returning `result` as-is: classifier.py/planner.py's
    # existing confidence=="low" and clarification_question check (after
    # call_role returns normally, no exception at all) halts for the real
    # question directly. When require_confirmation=False (set-and-forget,
    # no one to ask either way), this distinction is moot — fall through
    # to normal escalation as before, since there's no confirmation prompt
    # to have asked the wrong question via in the first place.
    has_clarification_question = bool(getattr(result, "clarification_question", None))
    if (
        escalated_to is None
        and allow_escalation
        and getattr(result, "confidence", None) == "low"
        and not (require_confirmation and has_clarification_question)
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


def call_role_with_repair(
    role:         str,
    repair_hint:  str = "Please output valid JSON.",
    max_attempts: int = 3,
    **call_role_kwargs,
) -> T:  # pyright: ignore[reportInvalidTypeVarUse]
    """
    call_role() wrapped in a retry loop for when the model's ENTIRE answer
    fails to parse — a malformed top-level payload, not just a field
    call_role's own Instructor-level repair already handles internally.
    Feeds the validation error back as a corrective follow-up turn and
    tries again. TruncatedOutputError is never retried here — it
    propagates immediately so pipeline/graph.py's generic truncation-retry
    wrapper can handle it (a token-cap truncation isn't a JSON problem;
    the same cap plus a "please output valid JSON" nudge won't help).

    repair_hint: role-specific guidance appended to the correction message
    (e.g. bugfix_node's reminder that search_text/replace_text must be
    valid escaped JSON strings). call_role_kwargs are passed straight
    through to call_role() — extra_messages, if given, seeds the first
    attempt and is then replaced by the correction turn on retries.
    """
    extra_messages = call_role_kwargs.pop("extra_messages", None) or []
    label = call_role_kwargs.get("stage", role)

    for attempt in range(max_attempts):
        try:
            return call_role(role, extra_messages=extra_messages or None, **call_role_kwargs)
        except TruncatedOutputError:
            raise
        except Exception as e:
            log.warning("%s: output validation failed (attempt %d/%d): %s", label, attempt + 1, max_attempts, e)
            if attempt == max_attempts - 1:
                raise RuntimeError(f"{label} failed to produce valid output after {max_attempts} attempts.") from e
            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {"role": "user",      "content": f"Your previous output failed validation:\n{e}\n\n{repair_hint}"},
            ]

    # Unreachable: the final attempt above either returns or raises.
    raise RuntimeError(f"{label}: call_role_with_repair exited retry loop unexpectedly")


def call_role_with_tool_history(*args, **kwargs) -> tuple[T, list["ToolCallRecord"]]:  # pyright: ignore[reportInvalidTypeVarUse]
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
    response_schema_arg = kwargs.get("response_schema")
    if response_schema_arg is None:
        raise ValueError(
            f"call_role_with_tool_history('{role}') requires response_schema= "
            f"(call_model_with_tools can't parse a final answer without one)."
        )
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
            response_schema = response_schema_arg,
            extra_messages  = kwargs.get("extra_messages"),
        )
    else:
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and prompt_def.get("system"):
            system_text = _safe_format(prompt_def["system"], kwargs.get("template_vars") or {})
            response_schema = response_schema_arg
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
        response_schema = response_schema_arg,
        stage           = stage,
        run_dir         = kwargs.get("run_dir", ""),
        thinking        = thinking,
        budget_tokens   = kwargs.get("budget_tokens"),
        max_tool_rounds = kwargs.get("max_tool_rounds", 4),
        max_retries     = kwargs.get("max_retries", 3),
        output_cap_override = kwargs.get("output_cap_override"),
    )