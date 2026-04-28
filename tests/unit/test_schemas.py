"""
tests/unit/test_schemas.py — validate all Pydantic schemas.

Tests that schemas accept valid data, reject invalid data,
and that enums and defaults behave correctly.
Run: pytest tests/unit/test_schemas.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pydantic import ValidationError

from schemas import (
    TaskClassification, Mode, TaskType, Complexity,
    VisualDescription,
    IdeationOutput, Approach,
    PlanSpec, ComponentSpec, FunctionSpec, Parameter,
    DraftOutput, ComponentDraft,
    AppraisalReport, Issue, IssueSeverity, IssueCategory,
    FixedOutput, AppliedFix,
    CritiqueVerdict, ValidationVerdict, CritiqueRecord,
    VerdictCategory, CriticScope,
    SubSpecInterface, InterfaceStatus,
    Lesson, LessonQuery,
)


# ── TaskClassification ────────────────────────────────────────────────────────

class TestTaskClassification:
    def test_valid_short_coding(self):
        tc = TaskClassification(
            mode="short", task_type="coding", complexity="simple",
            decompose=False, reasoning="Simple function request."
        )
        assert tc.mode == "short"
        assert tc.task_type == "coding"
        assert tc.decompose is False
        assert tc.estimated_sub_specs is None

    def test_valid_long_with_decompose(self):
        tc = TaskClassification(
            mode="long", task_type="mixed", complexity="complex",
            decompose=True, estimated_sub_specs=7,
            reasoning="Large multi-component system."
        )
        assert tc.decompose is True
        assert tc.estimated_sub_specs == 7

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            TaskClassification(
                mode="medium", task_type="coding", complexity="simple",
                decompose=False, reasoning="test"
            )

    def test_invalid_complexity(self):
        with pytest.raises(ValidationError):
            TaskClassification(
                mode="short", task_type="coding", complexity="extreme",
                decompose=False, reasoning="test"
            )

    def test_reasoning_required(self):
        with pytest.raises(ValidationError):
            TaskClassification(
                mode="short", task_type="coding", complexity="simple",
                decompose=False
            )


# ── PlanSpec ──────────────────────────────────────────────────────────────────

class TestPlanSpec:
    def _minimal_component(self, name="MyClass"):
        return ComponentSpec(
            name=name,
            responsibility="Does something",
        )

    def test_minimal_valid_plan(self):
        plan = PlanSpec(
            task_summary="Build a thing",
            chosen_approach="direct",
            components=[self._minimal_component()],
            implementation_order=["MyClass"],
            moe_routing_context="Python 3.11",
            confidence_in_plan="high",
        )
        assert len(plan.components) == 1
        assert plan.dropped_ideas == []
        assert plan.edge_cases == []

    def test_component_with_functions(self):
        func = FunctionSpec(
            name="process",
            description="Process input",
            parameters=[
                Parameter(name="data", type_hint="list[str]", description="Input data")
            ],
            returns="list[str]",
        )
        comp = ComponentSpec(
            name="Processor",
            responsibility="Processes things",
            functions=[func],
        )
        assert comp.functions[0].name == "process"
        assert comp.functions[0].parameters[0].required is True


# ── DraftOutput ───────────────────────────────────────────────────────────────

class TestDraftOutput:
    def test_valid_draft(self):
        draft = DraftOutput(
            component_drafts=[
                ComponentDraft(
                    component_name="MyClass",
                    code="class MyClass:\n    pass",
                )
            ]
        )
        assert len(draft.component_drafts) == 1
        assert draft.deviations_from_spec == []

    def test_empty_component_drafts(self):
        # Should be valid (guard checks this separately)
        draft = DraftOutput(component_drafts=[])
        assert draft.component_drafts == []


# ── AppraisalReport ───────────────────────────────────────────────────────────

class TestAppraisalReport:
    def test_valid_report_no_issues(self):
        report = AppraisalReport(
            overall_assessment="Looks good.",
            spec_satisfaction="high",
            critical_count=0,
            major_count=0,
            minor_count=0,
            components_reviewed=["MyClass"],
        )
        assert report.issues == []
        assert report.iq2s_inherited_issues == []

    def test_report_with_issues(self):
        issue = Issue(
            component="MyClass",
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.LOGIC_ERROR,
            description="Returns wrong value on edge case",
        )
        report = AppraisalReport(
            overall_assessment="Issues found.",
            spec_satisfaction="partial",
            issues=[issue],
            critical_count=1,
            major_count=0,
            minor_count=0,
            components_reviewed=["MyClass"],
        )
        assert report.critical_count == 1
        assert report.issues[0].severity == "critical"

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            Issue(
                component="X",
                severity="catastrophic",
                category=IssueCategory.LOGIC_ERROR,
                description="test",
            )


# ── ValidationVerdict ─────────────────────────────────────────────────────────

class TestValidationVerdict:
    def test_pass_verdict(self):
        v = ValidationVerdict(
            category="pass",
            synthesis_model="9b",
            description="Output satisfies the plan.",
        )
        assert v.category == "pass"
        assert v.specific_issues == []
        assert v.dissenting_notes is None

    def test_minor_fix_with_issues(self):
        v = ValidationVerdict(
            category="minor_fix",
            synthesis_model="9b",
            description="Small issues.",
            specific_issues=["Fix null check in process()", "Add missing import"],
        )
        assert len(v.specific_issues) == 2

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ValidationVerdict(
                category="unknown",
                synthesis_model="9b",
                description="test",
            )


# ── VerdictCategory enum ──────────────────────────────────────────────────────

class TestVerdictCategory:
    def test_all_values_defined(self):
        assert VerdictCategory.PASS         == "pass"
        assert VerdictCategory.MINOR_FIX    == "minor_fix"
        assert VerdictCategory.SPEC_PROBLEM == "spec_problem"
        assert VerdictCategory.UNRESOLVABLE == "unresolvable"


# ── SubSpecInterface ──────────────────────────────────────────────────────────

class TestSubSpecInterface:
    def test_valid_interface(self):
        iface = SubSpecInterface(
            sub_spec_uuid    = "abc-123",
            parent_run_uuid  = "parent-456",
            component_name   = "AuthModule",
            inputs           = ["UserCredentials"],
            outputs          = ["AuthToken"],
        )
        assert iface.status == "pending"
        assert iface.shared_object_refs == []


# ── Lesson ────────────────────────────────────────────────────────────────────

class TestLesson:
    def test_valid_lesson(self):
        lesson = Lesson(
            lesson_uuid      = "lesson-001",
            source_run_uuid  = "run-abc",
            issue_summary    = "Forgot await in async function",
            resolution_pattern="Add await before coroutine calls",
            task_type        = "coding",
            tags             = ["python", "async"],
            model_caught     = "deepcoder",
            issue_category   = "logic_error",
        )
        assert lesson.confidence_score == 1.0
        assert lesson.times_seen == 1
        assert lesson.is_meta_lesson is False

    def test_lesson_query_defaults(self):
        q = LessonQuery(task_type="coding", tags=["python"])
        assert q.top_k == 5
        assert q.min_score == 0.2
