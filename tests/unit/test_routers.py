"""
tests/unit/test_routers.py — routing function unit tests.

Routers are pure functions: state in → string out.
No model calls, no file I/O, no database.
Run: pytest tests/unit/test_routers.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from pipeline.routers import (
    route_after_input,
    route_after_classify,
    route_after_plan,
    route_after_draft_guard,
    route_after_draft_short_guard,
    route_after_bugfix,
    route_after_critic_a,
    route_after_validate,
)
from schemas.task_classification import TaskClassification
from schemas.validation import ValidationVerdict, VerdictCategory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classification(mode="short", task_type="coding", complexity="simple", decompose=False):
    return TaskClassification(
        mode=mode, task_type=task_type, complexity=complexity,
        decompose=decompose, reasoning="test"
    )

def _verdict(category: str) -> ValidationVerdict:
    return ValidationVerdict(
        category=category, synthesis_model="9b", description="test"
    )

def _state(**kwargs) -> dict:
    base = {
        "run_uuid":    "test-uuid",
        "run_dir":     "/tmp/test",
        "iteration":   0,
        "task_type":   "coding",
        "decompose":   False,
        "pipeline_complete": False,
        "pipeline_failed":   False,
    }
    base.update(kwargs)
    return base


# ── route_after_input ─────────────────────────────────────────────────────────

class TestRouteAfterInput:
    def test_no_image_goes_to_classify(self):
        state = _state()
        assert route_after_input(state) == "classify"

    def test_with_image_goes_to_vision_decode(self):
        state = _state(raw_image_path="/some/image.png")
        assert route_after_input(state) == "vision_decode"

    def test_empty_image_path_goes_to_classify(self):
        state = _state(raw_image_path="")
        assert route_after_input(state) == "classify"


# ── route_after_classify ──────────────────────────────────────────────────────

class TestRouteAfterClassify:
    def test_short_mode_goes_to_plan(self):
        state = _state(classification=_classification(mode="short"))
        assert route_after_classify(state) == "plan"

    def test_long_coding_goes_to_plan(self):
        state = _state(classification=_classification(mode="long", task_type="coding"))
        assert route_after_classify(state) == "plan"

    def test_long_ideation_goes_to_ideation(self):
        state = _state(classification=_classification(mode="long", task_type="ideation"))
        assert route_after_classify(state) == "ideation"

    def test_long_mixed_goes_to_ideation(self):
        state = _state(classification=_classification(mode="long", task_type="mixed"))
        assert route_after_classify(state) == "ideation"

    def test_no_classification_defaults_to_plan(self):
        state = _state()
        assert route_after_classify(state) == "plan"


# ── route_after_plan ──────────────────────────────────────────────────────────

class TestRouteAfterPlan:
    def test_long_mode_goes_to_draft(self):
        state = _state(classification=_classification(mode="long"), decompose=False)
        assert route_after_plan(state) == "draft"

    def test_short_mode_goes_to_draft_short(self):
        state = _state(classification=_classification(mode="short"), decompose=False)
        assert route_after_plan(state) == "draft_short"

    def test_decompose_true_goes_to_sub_spec_runner(self):
        state = _state(
            classification=_classification(mode="long"),
            decompose=True
        )
        assert route_after_plan(state) == "sub_spec_runner"

    def test_decompose_overrides_mode(self):
        # decompose=True should always go to sub_spec_runner regardless of mode
        state = _state(
            classification=_classification(mode="short"),
            decompose=True
        )
        assert route_after_plan(state) == "sub_spec_runner"


# ── route_after_draft_guard ───────────────────────────────────────────────────

class TestRouteAfterDraftGuard:
    def test_guard_passed_goes_to_appraise(self):
        state = _state(_guard_passed=True)
        assert route_after_draft_guard(state) == "appraise"

    def test_guard_failed_loops_to_draft(self):
        state = _state(_guard_passed=False)
        assert route_after_draft_guard(state) == "draft"

    def test_guard_missing_defaults_to_appraise(self):
        # Default is True (trusting)
        state = _state()
        assert route_after_draft_guard(state) == "appraise"


class TestRouteAfterDraftShortGuard:
    def test_guard_passed_goes_to_bugfix(self):
        state = _state(_guard_passed=True)
        assert route_after_draft_short_guard(state) == "bugfix"

    def test_guard_failed_loops_to_draft_short(self):
        state = _state(_guard_passed=False)
        assert route_after_draft_short_guard(state) == "draft_short"


# ── route_after_bugfix ────────────────────────────────────────────────────────

# Mock routing.yaml so tests don't require the file
_MOCK_ROUTING = {
    "ensemble": {
        "enabled": True,
        "trigger_on_complexity": ["complex"],
        "trigger_on_mode": ["long"],
        "run_critic_b": True,
        "force_on_final_iteration": True,
    },
    "correction_loop": {
        "max_iterations": 4,
        "minor_fix_target": {"long": "draft", "short": "bugfix"},
    },
}


class TestRouteAfterBugfix:
    def test_complex_long_goes_to_critic_a(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                classification=_classification(mode="long", complexity="complex"),
                iteration=0,
            )
            assert route_after_bugfix(state) == "critic_a"

    def test_simple_short_skips_ensemble(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                classification=_classification(mode="short", complexity="simple"),
                iteration=0,
            )
            assert route_after_bugfix(state) == "validate"

    def test_force_ensemble_on_final_iteration(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            # max_iterations=4, so final iteration is 3
            state = _state(
                classification=_classification(mode="short", complexity="simple"),
                iteration=3,
            )
            assert route_after_bugfix(state) == "critic_a"

    def test_ensemble_disabled_skips_all(self):
        cfg = dict(_MOCK_ROUTING)
        cfg["ensemble"] = dict(cfg["ensemble"])
        cfg["ensemble"]["enabled"] = False
        with patch("pipeline.routers._cfg", return_value=cfg):
            state = _state(
                classification=_classification(mode="long", complexity="complex"),
            )
            assert route_after_bugfix(state) == "validate"


# ── route_after_critic_a ──────────────────────────────────────────────────────

class TestRouteAfterCriticA:
    def test_critic_b_enabled_goes_to_critic_b(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state()
            assert route_after_critic_a(state) == "critic_b"

    def test_critic_b_disabled_goes_to_synthesise(self):
        cfg = dict(_MOCK_ROUTING)
        cfg["ensemble"] = dict(cfg["ensemble"])
        cfg["ensemble"]["run_critic_b"] = False
        with patch("pipeline.routers._cfg", return_value=cfg):
            state = _state()
            assert route_after_critic_a(state) == "synthesise"


# ── route_after_validate ──────────────────────────────────────────────────────

class TestRouteAfterValidate:
    def test_pass_goes_to_end(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("pass"),
                classification=_classification(mode="long"),
                iteration=1,
            )
            assert route_after_validate(state) == "end"

    def test_unresolvable_goes_to_end(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("unresolvable"),
                classification=_classification(mode="long"),
                iteration=1,
            )
            assert route_after_validate(state) == "end"

    def test_spec_problem_goes_to_plan(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("spec_problem"),
                classification=_classification(mode="long"),
                iteration=1,
            )
            assert route_after_validate(state) == "plan"

    def test_minor_fix_long_goes_to_draft(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("minor_fix"),
                classification=_classification(mode="long"),
                iteration=1,
            )
            assert route_after_validate(state) == "draft"

    def test_minor_fix_short_goes_to_bugfix(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("minor_fix"),
                classification=_classification(mode="short"),
                iteration=1,
            )
            assert route_after_validate(state) == "bugfix"

    def test_iteration_limit_forces_end(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state(
                validation_verdict=_verdict("minor_fix"),
                classification=_classification(mode="long"),
                iteration=4,   # at max_iterations
            )
            assert route_after_validate(state) == "end"

    def test_no_verdict_goes_to_end(self):
        with patch("pipeline.routers._cfg", return_value=_MOCK_ROUTING):
            state = _state()
            assert route_after_validate(state) == "end"
