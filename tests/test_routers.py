"""pipeline/routers.py — per-profile correction-loop cap, checked against YOUR real config/routing.yaml."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from pipeline import routers as R


@pytest.fixture(autouse=True)
def fresh_config_cache():
    R._routing_cfg.clear()
    yield
    R._routing_cfg.clear()


@pytest.fixture
def cfg():
    return R._cfg()                     # the real config/routing.yaml


def state(profile, category, iteration, complexity="simple"):
    return {"profile": profile, "iteration": iteration,
            "validation_verdict": NS(category=category),
            "classification": NS(complexity=complexity)}


def inject(**routing):
    R._routing_cfg.clear()
    R._routing_cfg.update(routing)
    return R._cfg()


# ── _max_iterations (pure config logic) ──────────────────────────────────────

def test_profile_key_overrides_the_global_cap():
    c = inject(correction_loop={"max_iterations": 4}, pipeline_profiles={"short": {"max_iterations": 0}})
    assert R._max_iterations(c, "short") == 0


def test_profiles_without_the_key_use_the_global_cap():
    c = inject(correction_loop={"max_iterations": 4}, pipeline_profiles={"short": {"max_iterations": 0}, "long": {}})
    assert R._max_iterations(c, "long") == 4
    assert R._max_iterations(c, "not-a-profile") == 4


def test_falls_back_to_4_when_nothing_is_configured():
    # plain dicts: _cfg() would reload the real file for an empty cache, but the helper takes cfg directly
    assert R._max_iterations({}, "short") == 4
    assert R._max_iterations({"pipeline_profiles": {"short": None}}, "short") == 4
    assert R._max_iterations({"pipeline_profiles": None}, "short") == 4


def test_cap_may_be_a_numeric_string():
    c = inject(pipeline_profiles={"short": {"max_iterations": "2"}})
    assert R._max_iterations(c, "short") == 2


# ── the real routing.yaml ────────────────────────────────────────────────────

def test_real_config_short_profile_gets_one_bugfix_pass(cfg):
    assert R._max_iterations(cfg, "short") == 0, "short should stop after draft -> ONE bugfix -> validate"


def test_real_config_other_profiles_still_loop(cfg):
    glob = cfg["correction_loop"]["max_iterations"]
    assert glob >= 1
    assert R._max_iterations(cfg, "medium") == glob
    assert R._max_iterations(cfg, "long") == glob


def test_real_config_ultra_honours_its_own_cap(cfg):
    """`ultra` declares max_iterations in routing.yaml; nothing read it before _max_iterations."""
    assert R._max_iterations(cfg, "ultra") == cfg["pipeline_profiles"]["ultra"]["max_iterations"]


# ── route_after_validate ─────────────────────────────────────────────────────

@pytest.mark.parametrize("iteration", [0, 1, 2, 5])
def test_short_stops_after_first_validate_whatever_the_verdict(cfg, iteration):
    for category in ("minor_fix", "pass", "unresolvable", "spec_problem"):
        assert R.route_after_validate(state("short", category, iteration)) == "distiller"


def test_medium_keeps_looping_until_the_cap(cfg):
    glob = cfg["correction_loop"]["max_iterations"]
    target = cfg["correction_loop"]["minor_fix_target"]["medium"]
    assert R.route_after_validate(state("medium", "minor_fix", glob - 1)) == target
    assert R.route_after_validate(state("medium", "minor_fix", glob)) == "distiller"


def test_long_redrafts_or_replans_below_the_cap(cfg):
    target = cfg["correction_loop"]["minor_fix_target"]["long"]
    assert R.route_after_validate(state("long", "minor_fix", 1)) == target
    assert R.route_after_validate(state("long", "spec_problem", 1)) == "plan"


def test_pass_and_unresolvable_end_the_run_on_every_profile(cfg):
    for profile in ("short", "medium", "long", "ultra"):
        assert R.route_after_validate(state(profile, "pass", 0)) == "distiller"
        assert R.route_after_validate(state(profile, "unresolvable", 0)) == "distiller"


def test_ultra_uses_its_higher_cap(cfg):
    ultra_cap = cfg["pipeline_profiles"]["ultra"]["max_iterations"]
    glob = cfg["correction_loop"]["max_iterations"]
    if ultra_cap <= glob:
        pytest.skip("ultra cap not above the global cap in this config")
    assert R.route_after_validate(state("ultra", "minor_fix", glob)) != "distiller"
    assert R.route_after_validate(state("ultra", "minor_fix", ultra_cap)) == "distiller"


# ── route_after_bugfix ───────────────────────────────────────────────────────

def test_short_always_validates_after_bugfix(cfg):
    assert R.route_after_bugfix(state("short", "x", 0)) == "validate"


def test_long_forces_the_ensemble_on_the_final_iteration(cfg):
    glob = cfg["correction_loop"]["max_iterations"]
    assert R.route_after_bugfix(state("long", "x", glob - 1, complexity="simple")) == "critic_a"
    assert R.route_after_bugfix(state("long", "x", 0, complexity="simple")) == "validate"
    assert R.route_after_bugfix(state("long", "x", 0, complexity="complex")) == "critic_a"


# ── other routers touched by the classify change ─────────────────────────────

def test_describe_tasks_bypass_plan_and_draft(cfg):
    s = {"profile": "short", "classification": NS(task_type="describe", complexity="simple"),
         "classifier_confidence": "high"}
    assert R.route_after_classify(s) == "describe"


def test_low_confidence_with_a_question_routes_to_clarify(cfg):
    s = {"profile": "short", "classification": NS(task_type="coding", complexity="simple"),
         "classifier_confidence": "low", "clarification_question": "Which cache?"}
    assert R.route_after_classify(s) == "clarify"


def test_coding_tasks_go_to_plan(cfg):
    s = {"profile": "short", "classification": NS(task_type="coding", complexity="simple"),
         "classifier_confidence": "high"}
    assert R.route_after_classify(s) == "plan"