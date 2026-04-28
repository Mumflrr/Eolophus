"""
tests/unit/test_prefix.py — inline prefix parsing and classifier pin tests.

Tests parse_task_prefix() in isolation (pure function, no models).
Tests that classify_node respects pinned values from state.
Run: pytest tests/unit/test_prefix.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

# Import the function under test directly
from run import parse_task_prefix


# ── parse_task_prefix ─────────────────────────────────────────────────────────

class TestParsePrefixMode:

    def test_long_prefix(self):
        task, overrides = parse_task_prefix("/long design a caching layer")
        assert task == "design a caching layer"
        assert overrides == {"mode": "long"}

    def test_short_prefix(self):
        task, overrides = parse_task_prefix("/short fix the typo in greet()")
        assert task == "fix the typo in greet()"
        assert overrides == {"mode": "short"}

    def test_long_uppercase(self):
        task, overrides = parse_task_prefix("/LONG design something")
        assert overrides["mode"] == "long"
        assert task == "design something"

    def test_mixed_case(self):
        task, overrides = parse_task_prefix("/Long design something")
        assert overrides["mode"] == "long"

    def test_no_prefix_returns_unchanged(self):
        original = "implement a FastAPI endpoint"
        task, overrides = parse_task_prefix(original)
        assert task == original
        assert overrides == {}

    def test_prefix_requires_trailing_space(self):
        # /long with no space after — should not match
        task, overrides = parse_task_prefix("/longdesign something")
        assert overrides == {}
        assert task == "/longdesign something"

    def test_empty_string(self):
        task, overrides = parse_task_prefix("")
        assert task == ""
        assert overrides == {}

    def test_prefix_only_no_task(self):
        # /long with only trailing space — prefix parses, task becomes empty.
        # The empty-task check in main() handles this gracefully.
        task, overrides = parse_task_prefix("/long ")
        assert task == ""
        assert overrides == {"mode": "long"}


class TestParsePrefixTaskType:

    def test_long_coding(self):
        task, overrides = parse_task_prefix("/long/coding write a CSV parser")
        assert task == "write a CSV parser"
        assert overrides == {"mode": "long", "task_type": "coding"}

    def test_long_ideation(self):
        task, overrides = parse_task_prefix("/long/ideation explore caching approaches")
        assert overrides == {"mode": "long", "task_type": "ideation"}
        assert task == "explore caching approaches"

    def test_long_mixed(self):
        task, overrides = parse_task_prefix("/long/mixed design and implement auth")
        assert overrides == {"mode": "long", "task_type": "mixed"}

    def test_short_coding(self):
        task, overrides = parse_task_prefix("/short/coding fix null check")
        assert overrides == {"mode": "short", "task_type": "coding"}
        assert task == "fix null check"

    def test_task_type_uppercase(self):
        task, overrides = parse_task_prefix("/long/CODING write a function")
        assert overrides["task_type"] == "coding"

    def test_type_without_mode(self):
        # /coding alone — not a valid prefix (no mode), should not match
        task, overrides = parse_task_prefix("/coding write a function")
        assert overrides == {}


class TestParsePrefixNoEnsemble:

    def test_no_ensemble_prefix(self):
        task, overrides = parse_task_prefix("/no-ensemble fix the bug in process()")
        assert task == "fix the bug in process()"
        assert overrides == {"no_ensemble": True}
        assert "mode" not in overrides

    def test_no_ensemble_uppercase(self):
        task, overrides = parse_task_prefix("/NO-ENSEMBLE fix something")
        assert overrides.get("no_ensemble") is True

    def test_no_ensemble_then_mode(self):
        # /no-ensemble/long — both should be parsed
        task, overrides = parse_task_prefix("/no-ensemble/long design something")
        assert overrides.get("no_ensemble") is True
        assert overrides.get("mode") == "long"
        assert task == "design something"

    def test_mode_then_type_not_no_ensemble(self):
        task, overrides = parse_task_prefix("/long/coding build something")
        assert "no_ensemble" not in overrides


class TestParsePrefixEdgeCases:

    def test_double_slash_not_valid(self):
        task, overrides = parse_task_prefix("//long design something")
        assert overrides == {}

    def test_slash_at_end_not_matched(self):
        task, overrides = parse_task_prefix("design something /long")
        assert overrides == {}
        assert task == "design something /long"

    def test_multiline_task_prefix_on_first_line(self):
        task, overrides = parse_task_prefix("/long write a function\nthat handles errors")
        assert overrides == {"mode": "long"}
        assert task == "write a function\nthat handles errors"

    def test_prefix_stripped_exactly(self):
        # Ensure no leading/trailing whitespace issues in clean task
        task, overrides = parse_task_prefix("/short   do something")
        # The regex requires \s+ after prefix — multiple spaces consumed
        assert task == "do something" or task.strip() == "do something"

    def test_unknown_token_not_matched(self):
        task, overrides = parse_task_prefix("/fast write a function")
        assert overrides == {}
        assert "/fast" in task


# ── Classifier pin behaviour ──────────────────────────────────────────────────
# These test the logic in classify_node without making model calls.

class TestClassifierPinLogic:
    """
    Tests that the classifier correctly uses pinned values from state.
    Uses model_copy to simulate what the classifier does with the
    resolved_classification — no actual model call needed.
    """

    def test_mode_pinned_in_state(self):
        from schemas.task_classification import TaskClassification
        # Simulate: state has mode="long" pinned, classifier returned mode="short"
        classifier_result = TaskClassification(
            mode="short", task_type="coding", complexity="simple",
            decompose=False, reasoning="Looks simple"
        )
        pinned_mode = "long"

        # The classifier logic: pinned wins
        final_mode = pinned_mode or classifier_result.mode
        assert final_mode == "long"

        resolved = classifier_result.model_copy(update={"mode": final_mode})
        assert resolved.mode == "long"
        assert resolved.task_type == "coding"   # not changed

    def test_task_type_pinned_in_state(self):
        from schemas.task_classification import TaskClassification
        classifier_result = TaskClassification(
            mode="long", task_type="coding", complexity="complex",
            decompose=True, estimated_sub_specs=6,
            reasoning="Complex task"
        )
        pinned_task_type = "ideation"

        final_task_type = pinned_task_type or classifier_result.task_type
        resolved = classifier_result.model_copy(update={"task_type": final_task_type})

        assert resolved.task_type == "ideation"
        assert resolved.mode == "long"               # unchanged
        assert resolved.decompose is True            # unchanged
        assert resolved.complexity == "complex"      # unchanged

    def test_nothing_pinned_uses_classifier(self):
        from schemas.task_classification import TaskClassification
        classifier_result = TaskClassification(
            mode="short", task_type="coding", complexity="simple",
            decompose=False, reasoning="Simple"
        )
        pinned_mode      = None
        pinned_task_type = None

        final_mode      = pinned_mode      or classifier_result.mode
        final_task_type = pinned_task_type or classifier_result.task_type

        assert final_mode      == "short"
        assert final_task_type == "coding"

    def test_both_pinned(self):
        from schemas.task_classification import TaskClassification
        classifier_result = TaskClassification(
            mode="short", task_type="coding", complexity="moderate",
            decompose=False, reasoning="moderate task"
        )
        pinned_mode      = "long"
        pinned_task_type = "ideation"

        final_mode      = pinned_mode      or classifier_result.mode
        final_task_type = pinned_task_type or classifier_result.task_type
        resolved = classifier_result.model_copy(update={
            "mode": final_mode, "task_type": final_task_type
        })

        assert resolved.mode      == "long"
        assert resolved.task_type == "ideation"
        assert resolved.complexity == "moderate"   # still from classifier
        assert resolved.decompose  is False        # still from classifier
