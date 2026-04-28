"""
tests/unit/test_guards.py — deterministic guard logic tests.

Guards have no model calls and are purely deterministic —
ideal for fast, reliable unit tests.
Run: pytest tests/unit/test_guards.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from pipeline.guards import (
    check_lazy_evaluation,
    check_fixed_output_present,
    check_iteration_limit,
    check_interface_compatibility,
)
from schemas.execution import DraftOutput, ComponentDraft, FixedOutput, AppliedFix
from schemas.sub_spec import SubSpecInterface, InterfaceStatus


# ── check_lazy_evaluation ─────────────────────────────────────────────────────

class TestLazyEvaluationGuard:

    def _draft(self, code: str, notes: str = "") -> DraftOutput:
        return DraftOutput(
            component_drafts=[ComponentDraft(component_name="MyComp", code=code)],
            implementation_notes=notes or None,
        )

    def test_passes_real_python_function(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "def process(data: list[str]) -> list[str]:\n    return [x.upper() for x in data]"
        ))
        assert passed is True
        assert reason == "ok"

    def test_passes_real_class(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "class DataProcessor:\n    def __init__(self):\n        self.cache = {}\n\n"
            "    def run(self, x: int) -> int:\n        return x * 2"
        ))
        assert passed is True

    def test_passes_async_function(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as s:\n"
            "        resp = await s.get(url)\n        return await resp.json()"
        ))
        assert passed is True

    def test_fails_empty_code(self):
        passed, reason = check_lazy_evaluation(self._draft(""))
        assert passed is False
        assert "empty" in reason.lower()

    def test_fails_no_component_drafts(self):
        draft = DraftOutput(component_drafts=[])
        passed, reason = check_lazy_evaluation(draft)
        assert passed is False

    def test_fails_lazy_phrase_already_implemented(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "The code is already implemented correctly."
        ))
        assert passed is False
        assert "lazy" in reason.lower() or "phrase" in reason.lower()

    def test_fails_lazy_phrase_no_changes_needed(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "No changes are needed to the existing implementation."
        ))
        assert passed is False

    def test_fails_lazy_phrase_already_exists(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "The function already exists and works correctly."
        ))
        assert passed is False

    def test_fails_prose_only_no_code_indicators(self):
        passed, reason = check_lazy_evaluation(self._draft(
            "This would involve creating a service layer that handles "
            "the business logic for user authentication."
        ))
        assert passed is False

    def test_fails_lazy_in_implementation_notes(self):
        passed, reason = check_lazy_evaluation(
            self._draft(
                "def greet(name: str) -> str:\n    return f'Hello, {name}'",
                notes="The implementation is already complete and needs no changes."
            )
        )
        assert passed is False

    def test_passes_import_only_is_minimal_but_ok(self):
        # import statements are valid code indicators
        passed, reason = check_lazy_evaluation(self._draft(
            "import os\nimport sys\nfrom pathlib import Path"
        ))
        assert passed is True

    def test_handles_multiple_components(self):
        draft = DraftOutput(component_drafts=[
            ComponentDraft(
                component_name="CompA",
                code="class CompA:\n    pass"
            ),
            ComponentDraft(
                component_name="CompB",
                code="def run_b() -> None:\n    pass"
            ),
        ])
        passed, reason = check_lazy_evaluation(draft)
        assert passed is True


# ── check_fixed_output_present ────────────────────────────────────────────────

class TestFixedOutputGuard:

    def test_passes_valid_fixed_output(self):
        fixed = FixedOutput(
            component_drafts=[
                ComponentDraft(component_name="MyComp", code="def f(): pass")
            ],
            overall_quality="good",
        )
        passed, reason = check_fixed_output_present(fixed)
        assert passed is True

    def test_fails_empty_component_drafts(self):
        fixed = FixedOutput(component_drafts=[], overall_quality="unknown")
        passed, reason = check_fixed_output_present(fixed)
        assert passed is False

    def test_fails_all_empty_code(self):
        fixed = FixedOutput(
            component_drafts=[
                ComponentDraft(component_name="X", code=""),
                ComponentDraft(component_name="Y", code="   "),
            ],
            overall_quality="unknown",
        )
        passed, reason = check_fixed_output_present(fixed)
        assert passed is False


# ── check_iteration_limit ─────────────────────────────────────────────────────

class TestIterationLimitGuard:

    def _state(self, iteration: int) -> dict:
        return {"iteration": iteration, "run_uuid": "test-uuid", "run_dir": "/tmp"}

    def test_within_limit(self):
        passed, reason = check_iteration_limit(self._state(0), max_iterations=4)
        assert passed is True

    def test_within_limit_last_iteration(self):
        passed, reason = check_iteration_limit(self._state(3), max_iterations=4)
        assert passed is True

    def test_at_limit(self):
        passed, reason = check_iteration_limit(self._state(4), max_iterations=4)
        assert passed is False
        assert "4/4" in reason

    def test_over_limit(self):
        passed, reason = check_iteration_limit(self._state(10), max_iterations=4)
        assert passed is False

    def test_zero_max_iterations(self):
        passed, reason = check_iteration_limit(self._state(0), max_iterations=0)
        assert passed is False


# ── check_interface_compatibility ─────────────────────────────────────────────

class TestInterfaceCompatibility:

    def _iface(self, name, inputs, outputs, refs=None) -> SubSpecInterface:
        from schemas.sub_spec import SharedObjectRef
        shared = []
        if refs:
            for r in refs:
                shared.append(SharedObjectRef(
                    name=r, defined_in="root", type_hint="Any", description="shared"
                ))
        return SubSpecInterface(
            sub_spec_uuid   = f"uuid-{name}",
            parent_run_uuid = "parent",
            component_name  = name,
            inputs          = inputs,
            outputs         = outputs,
            shared_object_refs = shared,
        )

    def test_compatible_simple(self):
        ifaces = [
            self._iface("A", inputs=[],          outputs=["UserToken"]),
            self._iface("B", inputs=["UserToken"],outputs=["Response"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is True
        assert violations == []

    def test_compatible_no_dependencies(self):
        ifaces = [
            self._iface("A", inputs=[], outputs=["X"]),
            self._iface("B", inputs=[], outputs=["Y"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is True

    def test_missing_input_provider(self):
        ifaces = [
            self._iface("A", inputs=["MissingType"], outputs=["X"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is False
        assert any("MissingType" in v for v in violations)

    def test_missing_shared_object(self):
        ifaces = [
            self._iface("A", inputs=[], outputs=["X"], refs=["SharedConfig"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is False
        assert any("SharedConfig" in v for v in violations)

    def test_shared_object_defined_in_outputs(self):
        ifaces = [
            self._iface("A", inputs=[], outputs=["SharedConfig", "X"]),
            self._iface("B", inputs=[], outputs=["Y"], refs=["SharedConfig"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is True

    def test_empty_interfaces(self):
        ok, violations = check_interface_compatibility([])
        assert ok is True
        assert violations == []

    def test_multiple_violations(self):
        ifaces = [
            self._iface("A", inputs=["Missing1", "Missing2"], outputs=["X"]),
        ]
        ok, violations = check_interface_compatibility(ifaces)
        assert ok is False
        assert len(violations) >= 2
