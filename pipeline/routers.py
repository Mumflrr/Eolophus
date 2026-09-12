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


# ── Profile resolution ────────────────────────────────────────────────────────
#
# Replaces classification.mode ("short"|"long") as the thing routers consult
# for pipeline SHAPE. mode remains on TaskClassification as an informational
# hint (nodes/classifier.py still fills it in) but is no longer read by any
# router below except as a legacy fallback when state["profile"] is somehow
# absent (should not happen on any run that went through classify_node with
# this file's route_after_classify, but avoids a hard crash on old
# checkpoints resumed against new code).

_VALID_PROFILES = ("short", "medium", "long", "ultra")


def select_profile(classification) -> str:
    """
    Resolve requested_profile="auto" to a concrete profile name, using
    ONLY TaskClassification's own output — no separate heuristic table.
    Called once, right after classify_node, before route_after_classify
    decides where to go next.

    Rule (per design doc §4 item 2 — "does profile selection even need to
    be smarter than picking off the classifier's own fields?"):
      decompose=True                          → "long"   (needs sub_spec_runner,
                                                            which only long/ultra have)
      complexity=="complex"                   → "long"
      complexity=="moderate"                  → "medium"
      complexity=="simple"                    → "short"
    task_type=="describe" is not special-cased here — route_after_classify
    sends describe tasks straight to describe_node regardless of profile,
    so profile selection doesn't need to account for it.
    """
    if classification is None:
        return "short"

    if getattr(classification, "decompose", False):
        return "long"

    complexity = getattr(classification, "complexity", "simple")
    if complexity == "complex":
        return "long"
    if complexity == "moderate":
        return "medium"
    return "short"


def resolve_profile(state: PipelineState) -> str:
    """
    Return the ALREADY-RESOLVED profile for this run. Auto-resolution
    itself happens once, in classify_node (via select_profile above) —
    this function just reads state["profile"] back, with a legacy
    fallback onto classification.mode for states that predate the
    profile field.
    """
    profile = state.get("profile")
    if profile in _VALID_PROFILES:
        return profile

    # Legacy fallback — old checkpoint or a state that skipped classify_node
    classification = state.get("classification")
    mode = getattr(classification, "mode", "short") if classification else "short"
    log.warning(
        "Router: state['profile'] missing/invalid (%r) — falling back to "
        "legacy mode-based inference (mode=%s)", profile, mode,
    )
    return "long" if mode == "long" else "short"


def node_set_for(profile: str) -> list[str]:
    """Return the configured node_set list for a profile name."""
    cfg = _cfg()
    profiles = cfg.get("pipeline_profiles", {})
    return profiles.get(profile, {}).get("node_set", [])


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
    After planning: select executor based on the active profile's node_set
    and the decompose flag.
    decompose=True                    → sub_spec_runner (top-level runs only,
                                          and only if the profile's node_set
                                          actually includes it — short/medium
                                          don't)
    profile's node_set has "draft"    → draft (35B / whatever escalated to)
    otherwise (node_set has "draft_short") → draft_short

    Sub-spec runs (is_sub_spec=True) never decompose further — this prevents
    infinite recursion when a sub-spec task is misclassified as complex.

    plan_node can also halt for clarification on its own (independent of
    classify's confidence check) — see PlanSpec.confidence /
    plan.clarification_question in nodes/planner.py. When it does, it
    returns pipeline_halted=True and skips writing plan_spec entirely, so
    this MUST be checked first — otherwise draft/draft_short run next and
    crash on a missing plan_spec.
    """
    if state.get("pipeline_halted"):
        log.debug("Router: plan → clarify (pipeline_halted)")
        return "clarify"

    profile   = resolve_profile(state)
    node_set  = node_set_for(profile)
    is_sub    = state.get("is_sub_spec", False)
    decompose = state.get("decompose", False) and not is_sub and "sub_spec_runner" in node_set

    if decompose:
        log.debug("Router: plan → sub_spec_runner (profile=%s)", profile)
        return "sub_spec_runner"

    if "draft" in node_set:
        log.debug("Router: plan → draft (profile=%s)", profile)
        return "draft"

    log.debug("Router: plan → draft_short (profile=%s)", profile)
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
    """
    After guard on the draft_short path (short and medium profiles both
    route here — see their node_set in routing.yaml's pipeline_profiles):
    → bugfix (skipping appraisal), or loop back to draft_short on guard
    failure. No profile branching needed — this router is only ever
    reached from draft_short, whichever profile got there.
    """
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
      - the active profile's node_set has no critic_a/critic_b nodes at
        all (short/medium — see routing.yaml's pipeline_profiles)

    REMOVED: the old PIPELINE_NO_ENSEMBLE env var / --no-ensemble flag /
    /no-ensemble chat prefix. That was a manual per-run override sitting
    alongside profile selection as a SECOND way to decide the same thing
    ensemble-or-not now already decides via node_set — a short/medium run
    never had ensemble nodes to route to regardless of the env var, and a
    long/ultra run wanting to skip ensemble should just request the
    short/medium profile instead, rather than keeping two independent
    knobs that could disagree with each other. See design doc's
    accompanying note on this removal.
    """
    cfg = _cfg()
    ensemble_cfg  = cfg.get("ensemble", {})

    if not ensemble_cfg.get("enabled", True):
        return "validate"

    classification = state.get("classification")
    complexity = getattr(classification, "complexity", "simple") if classification else "simple"
    profile    = resolve_profile(state)
    node_set   = node_set_for(profile)
    iteration  = state.get("iteration", 0)
    max_iter   = cfg.get("correction_loop", {}).get("max_iterations", 4)

    # Ensemble nodes (critic_a/critic_b) only exist in long/ultra's node_set
    # — short/medium have nowhere to route an ensemble trigger to, so treat
    # them as never-triggering regardless of what trigger_on_profile says.
    if "critic_a" not in node_set:
        log.debug("Router: bugfix → validate (profile=%s has no ensemble nodes)", profile)
        return "validate"

    trigger_complexities = ensemble_cfg.get("trigger_on_complexity", ["complex"])
    trigger_profiles     = ensemble_cfg.get("trigger_on_profile",    ["long", "ultra"])
    force_final          = ensemble_cfg.get("force_on_final_iteration", True)

    # Force ensemble on last iteration before giving up
    if force_final and iteration >= max_iter - 1:
        log.debug("Router: bugfix → critic_a (forced final iteration)")
        return "critic_a"

    if complexity in trigger_complexities and profile in trigger_profiles:
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

    pass            → distiller (to learn from success)
    minor_fix       → draft (long) or bugfix (short) depending on mode
    spec_problem    → plan
    unresolvable    → distiller (to learn from failure)
    """
    verdict = state.get("validation_verdict")
    if not verdict:
        log.warning("Router: no validation verdict found → distiller")
        return "distiller"

    category   = getattr(verdict, "category", "unresolvable")
    iteration  = state.get("iteration", 0)
    cfg        = _cfg()
    max_iter   = cfg.get("correction_loop", {}).get("max_iterations", 4)
    profile    = resolve_profile(state)

    log.info("Router: verdict=%s iter=%d/%d profile=%s", category, iteration, max_iter, profile)

    if category == "pass":
        return "distiller"

    if category == "unresolvable" or iteration >= max_iter:
        return "distiller"

    if category == "spec_problem":
        return "plan"

    if category == "minor_fix":
        # long/ultra: full redraft by 35B (or whatever escalated to)
        # short/medium: targeted fix by coder14b
        fix_targets = cfg.get("correction_loop", {}).get("minor_fix_target", {})
        # ultra shares long's node_set/redraft target; falls back to "long"'s
        # entry rather than needing its own duplicate key in routing.yaml
        lookup_key = "long" if profile == "ultra" else profile
        target = fix_targets.get(lookup_key, "bugfix")
        return target

    # Fallback
    log.warning("Router: unknown verdict category '%s' → distiller", category)
    return "distiller"


# ── Sub-spec and final validation ─────────────────────────────────────────────

def route_after_sub_specs(state: PipelineState) -> str:
    """After all sub-specs complete: → final_validate."""
    return "final_validate"


def route_after_final_validate(state: PipelineState) -> str:
    """
    After final validation: route to distiller to extract lessons 
    before ending the pipeline.
    """
    return "distiller"


def route_after_classify_with_confidence(state) -> str:
    """
    Extended route_after_classify that checks confidence first.
    Drop-in replacement for route_after_classify in graph.py.

    Routes:
      confidence=low + question → "clarify"
      profile has "ideation" in its node_set + task_type in (ideation,mixed) → "ideation"
      all others                → "plan"

    On confidence=low reaching this router at all: clients/llm.py's
    call_role() already tried escalating classify up its
    escalation_ladders entry when TaskClassification.confidence=="low" (see
    design doc §2.3/§2.6) before classify_node ever returned. Reaching here
    with confidence still "low" means the ladder was exhausted with no
    improvement — in human_in_the_loop mode that's a genuine halt; in
    set-and-forget mode classify_node/classifier.py already downgrades this
    itself (proceeds best-effort) rather than setting clarification_question,
    so "confidence == low and question present" should only actually occur
    here for human_in_the_loop runs.
    """
    import logging
    log = logging.getLogger(__name__)

    confidence = state.get("classifier_confidence", "high")
    question   = state.get("clarification_question")

    if confidence == "low" and question:
        log.info("Router: classify → clarify (confidence=low after escalation exhausted)")
        return "clarify"

    classification = state.get("classification")
    if not classification:
        return "plan"

    profile   = resolve_profile(state)
    task_type = getattr(classification, "task_type", "coding")

    if "ideation" in node_set_for(profile) and task_type in ("ideation", "mixed"):
        return "ideation"

    return "plan"


def route_after_classify(state) -> str:
    """
    Main post-classify router. Checks confidence, then task_type.
    Routes describe tasks directly to describe_node, bypassing plan/draft.

    See route_after_classify_with_confidence's docstring above for why
    confidence=="low" reaching this point means escalation already ran
    and was exhausted, not that escalation hasn't happened yet.
    """
    import logging
    log = logging.getLogger(__name__)

    confidence = state.get("classifier_confidence", "high")
    question   = state.get("clarification_question")
    if confidence == "low" and question:
        log.info("Router: classify → clarify (confidence=low after escalation exhausted)")
        return "clarify"

    classification = state.get("classification")
    if not classification:
        return "plan"

    profile   = resolve_profile(state)
    task_type = getattr(classification, "task_type", "coding")

    # Describe tasks bypass the entire plan/draft pipeline
    if task_type == "describe":
        log.info("Router: classify → describe (task_type=describe)")
        return "describe"

    # ideation fires only if the active profile's node_set actually has it
    if "ideation" in node_set_for(profile) and task_type in ("ideation", "mixed"):
        return "ideation"

    return "plan"