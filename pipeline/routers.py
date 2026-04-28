"""
pipeline/routers.py — all LangGraph conditional edge functions.

Each router receives the full PipelineState and returns a string
identifying the next node. No routing logic lives anywhere else.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from pipeline.state import PipelineState

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_routing_cfg: dict = {}

def _cfg() -> dict:
    if not _routing_cfg:
        p = Path(__file__).parent.parent / "config" / "routing.yaml"
        with open(p) as f:
            _routing_cfg.update(yaml.safe_load(f))
    return _routing_cfg


# ── Entry routers ─────────────────────────────────────────────────────────────

def route_after_input(state: PipelineState) -> str:
    """
    After input is received: does it contain an image?
    YES → vision_decode
    NO  → classify
    """
    if state.get("raw_image_path"):
        log.debug("Router: input → vision_decode")
        return "vision_decode"
    log.debug("Router: input → classify")
    return "classify"


def route_after_vision(state: PipelineState) -> str:
    """After vision decode: always → classify."""
    return "classify"


def route_after_ideation(state: PipelineState) -> str:
    """After ideation: always → plan."""
    return "plan"


# ── Execution router ──────────────────────────────────────────────────────────

def route_after_plan(state: PipelineState) -> str:
    """
    After planning: select executor based on mode and decompose flag.
    decompose=True → sub_spec_runner
    long mode      → draft (35B)
    short mode     → draft_short (9B)
    """
    classification = state.get("classification")
    mode    = getattr(classification, "mode", "short") if classification else "short"
    decompose = state.get("decompose", False)

    if decompose:
        log.debug("Router: plan → sub_spec_runner")
        return "sub_spec_runner"

    if mode == "long":
        log.debug("Router: plan → draft")
        return "draft"

    log.debug("Router: plan → draft_short")
    return "draft_short"


# ── Guard router ──────────────────────────────────────────────────────────────

def route_after_draft_guard(state: PipelineState) -> str:
    """
    After lazy evaluation guard:
    guard_passed=True  → appraise
    guard_passed=False → draft (loop back)
    """
    if state.get("_guard_passed", True):
        return "appraise"
    log.warning("Router: lazy eval guard failed → redraft")
    return "draft"


def route_after_draft_short_guard(state: PipelineState) -> str:
    """Short mode: after guard → bugfix (skipping appraisal)."""
    if state.get("_guard_passed", True):
        return "bugfix"
    return "draft_short"


# ── Post-fix router ───────────────────────────────────────────────────────────

def route_after_bugfix(state: PipelineState) -> str:
    """
    After bug fix pass:
    ensemble enabled + complexity triggers it → critic_a
    otherwise                                 → validate

    Ensemble is suppressed when:
      - routing.yaml ensemble.enabled is False
      - PIPELINE_NO_ENSEMBLE env var is set (from --no-ensemble flag or /no-ensemble prefix)
    """
    import os

    # Check env var set by --no-ensemble flag or /no-ensemble prefix
    if os.environ.get("PIPELINE_NO_ENSEMBLE"):
        log.debug("Router: bugfix → validate (PIPELINE_NO_ENSEMBLE set)")
        return "validate"

    cfg = _cfg()
    ensemble_cfg  = cfg.get("ensemble", {})

    if not ensemble_cfg.get("enabled", True):
        return "validate"

    classification = state.get("classification")
    complexity = getattr(classification, "complexity", "simple") if classification else "simple"
    mode       = getattr(classification, "mode",       "short")  if classification else "short"
    iteration  = state.get("iteration", 0)
    max_iter   = cfg.get("correction_loop", {}).get("max_iterations", 4)

    trigger_complexities = ensemble_cfg.get("trigger_on_complexity", ["complex"])
    trigger_modes        = ensemble_cfg.get("trigger_on_mode",        ["long"])
    force_final          = ensemble_cfg.get("force_on_final_iteration", True)

    # Force ensemble on last iteration before giving up
    if force_final and iteration >= max_iter - 1:
        log.debug("Router: bugfix → critic_a (forced final iteration)")
        return "critic_a"

    if complexity in trigger_complexities and mode in trigger_modes:
        log.debug("Router: bugfix → critic_a")
        return "critic_a"

    log.debug("Router: bugfix → validate (skipping ensemble)")
    return "validate"


# ── Ensemble routers ──────────────────────────────────────────────────────────

def route_after_critic_a(state: PipelineState) -> str:
    """After Critic A: run Critic B if enabled, else synthesise."""
    cfg = _cfg()
    if cfg.get("ensemble", {}).get("run_critic_b", True):
        return "critic_b"
    return "synthesise"


def route_after_critic_b(state: PipelineState) -> str:
    """After Critic B: always → synthesise."""
    return "synthesise"


def route_after_synthesise(state: PipelineState) -> str:
    """After synthesis: always → validate."""
    return "validate"


# ── Verdict router ────────────────────────────────────────────────────────────

def route_after_validate(state: PipelineState) -> str:
    """
    Core routing decision based on ValidationVerdict.category.

    pass            → end
    minor_fix       → draft (long) or bugfix (short) depending on mode
    spec_problem    → plan
    unresolvable    → end (with failure flag)
    """
    verdict = state.get("validation_verdict")
    if not verdict:
        log.warning("Router: no validation verdict found → end")
        return "end"

    category   = getattr(verdict, "category", "unresolvable")
    iteration  = state.get("iteration", 0)
    cfg        = _cfg()
    max_iter   = cfg.get("correction_loop", {}).get("max_iterations", 4)
    classification = state.get("classification")
    mode       = getattr(classification, "mode", "short") if classification else "short"

    log.info("Router: verdict=%s iter=%d/%d mode=%s", category, iteration, max_iter, mode)

    if category == "pass":
        return "end"

    if category == "unresolvable" or iteration >= max_iter:
        return "end"

    if category == "spec_problem":
        return "plan"

    if category == "minor_fix":
        # Long mode: full redraft by 35B
        # Short mode: targeted fix by coder14b
        fix_targets = cfg.get("correction_loop", {}).get("minor_fix_target", {})
        target = fix_targets.get(mode, "bugfix")
        return target

    # Fallback
    log.warning("Router: unknown verdict category '%s' → end", category)
    return "end"


# ── Sub-spec and final validation ─────────────────────────────────────────────

def route_after_sub_specs(state: PipelineState) -> str:
    """After all sub-specs complete: → final_validate."""
    return "final_validate"


def route_after_final_validate(state: PipelineState) -> str:
    """After final validation: always → end."""
    return "end"


def route_after_classify_with_confidence(state) -> str:
    """
    Extended route_after_classify that checks confidence first.
    Drop-in replacement for route_after_classify in graph.py.
 
    Routes:
      confidence=low + question → "clarify"
      long + ideation/mixed     → "ideation"
      all others                → "plan"
    """
    import logging
    log = logging.getLogger(__name__)
 
    # Halt on low confidence before spending any model time
    confidence = state.get("classifier_confidence", "high")
    question   = state.get("clarification_question")
 
    if confidence == "low" and question:
        log.info("Router: classify → clarify (confidence=low, question present)")
        return "clarify"
 
    classification = state.get("classification")
    if not classification:
        return "plan"
 
    mode      = getattr(classification, "mode",      "short")
    task_type = getattr(classification, "task_type", "coding")
 
    if mode == "long" and task_type in ("ideation", "mixed"):
        return "ideation"
 
    return "plan"
 
 
def route_after_classify(state) -> str:
    """
    Main post-classify router. Checks confidence, then task_type.
    Routes describe tasks directly to describe_node, bypassing plan/draft.
    """
    import logging
    log = logging.getLogger(__name__)
 
    # Halt on low confidence before spending any model time
    confidence = state.get("classifier_confidence", "high")
    question   = state.get("clarification_question")
    if confidence == "low" and question:
        log.info("Router: classify → clarify (confidence=low)")
        return "clarify"
 
    classification = state.get("classification")
    if not classification:
        return "plan"
 
    mode      = getattr(classification, "mode",      "short")
    task_type = getattr(classification, "task_type", "coding")
 
    # Describe tasks bypass the entire plan/draft pipeline
    if task_type == "describe":
        log.info("Router: classify → describe (task_type=describe)")
        return "describe"
 
    # Long ideation/mixed fires 27B
    if mode == "long" and task_type in ("ideation", "mixed"):
        return "ideation"
 
    return "plan"