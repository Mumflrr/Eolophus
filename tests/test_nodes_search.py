"""nodes/planner.py, nodes/ideation.py, nodes/drafter.py — how search flows through the pipeline."""
from __future__ import annotations

import pytest

from clients import tools
from clients.llm import ToolCallRecord
from nodes import drafter, ideation, planner
from tests.fakes import FakePydantic


def rec(query, result):
    return ToolCallRecord(name="search_web", arguments={"query": query}, result=result)


@pytest.fixture
def role(monkeypatch):
    """
    Replace call_role in all three nodes with a scripted fake. Set role.history to
    what the "model" searched on the next call; role.calls holds every call's kwargs.
    """
    box = type("Box", (), {})()
    box.calls, box.history, box.result = [], [], FakePydantic()

    def fake_call_role(**kw):
        box.calls.append(kw)
        if kw.get("tools") and kw.get("tool_history_sink") is not None:
            kw["tool_history_sink"][:] = list(box.history)
        return box.result

    for mod in (planner, ideation, drafter):
        monkeypatch.setattr(mod, "call_role", fake_call_role)
    # planner's other collaborators
    monkeypatch.setattr(planner, "retrieve_lessons", lambda **kw: [])
    monkeypatch.setattr(planner, "format_lessons_for_prompt", lambda l: "")
    monkeypatch.setattr(planner, "LessonQuery", lambda **kw: kw)
    monkeypatch.setattr(planner, "compress_text", lambda t, **kw: t)
    # drafter's collaborators
    monkeypatch.setattr(drafter, "check_lazy_evaluation", lambda d: (True, ""))
    monkeypatch.setattr(drafter, "check_ast_syntax", lambda c: (True, ""))
    monkeypatch.setattr(drafter, "write_iteration_artifact", lambda rd, fn, content, it: f"{rd}/{fn}")
    return box


def st(run_dir, **kw):
    return {"run_dir": run_dir, "raw_text_input": "What's the latest FastAPI?", **kw}


# ── plan_node ────────────────────────────────────────────────────────────────

def test_plan_search_off_offers_no_tool_and_no_hint(role, run_dir):
    out = planner.plan_node(st(run_dir, use_search=False))
    kw = role.calls[0]
    assert kw["tools"] is None and kw["tool_impls"] is None
    assert kw["template_vars"]["search_hint"] == "" and kw["template_vars"]["chat_block"] == ""
    assert out["search_notes"] == ""


def test_plan_search_on_offers_tool_hint_and_sink(role, run_dir):
    role.history = [rec("fastapi latest", "[Web search results]\n1. FastAPI 0.115.2")]
    out = planner.plan_node(st(run_dir, use_search=True))
    kw = role.calls[0]
    assert kw["tools"] == [tools.SEARCH_TOOL_SCHEMA]
    assert kw["tool_impls"] is tools.TOOL_IMPLEMENTATIONS
    assert kw["template_vars"]["search_hint"] == tools.SEARCH_HINT
    assert isinstance(kw["tool_history_sink"], list)
    assert "FastAPI 0.115.2" in out["search_notes"] and "fastapi latest" in out["search_notes"]


def test_plan_search_on_but_model_did_not_search(role, run_dir):
    assert planner.plan_node(st(run_dir, use_search=True))["search_notes"] == ""


def test_plan_overwrites_stale_notes_on_a_fresh_turn(role, run_dir):
    """A chat follow-up reuses the checkpoint; iteration restarts at 0 and old notes must not survive."""
    out = planner.plan_node(st(run_dir, use_search=True, iteration=0, search_notes="STALE from last turn"))
    assert out["search_notes"] == ""


def test_plan_replan_keeps_earlier_notes(role, run_dir):
    out = planner.plan_node(st(run_dir, use_search=True, iteration=2, search_notes="Query: first\nkept"))
    assert "kept" in out["search_notes"]


def test_plan_combines_ideation_and_plan_notes(role, run_dir):
    role.history = [rec("plan-side", "plan result")]
    out = planner.plan_node(st(run_dir, use_search=True, ideation_output=FakePydantic(),
                               search_notes="Query: idea-side\nideation result"))
    notes = out["search_notes"]
    assert notes.index("ideation result") < notes.index("plan result")


def test_plan_writes_planspec_and_clears_ideation(role, run_dir):
    out = planner.plan_node(st(run_dir))
    assert out["ideation_output"] is None
    assert out["plan_spec_path"].endswith("planspec.json")
    assert "plan_spec" in out and "relevant_lessons" in out


def test_plan_low_confidence_halts_when_human_in_the_loop(role, run_dir):
    role.result = FakePydantic(confidence="low", clarification_question="Which cache?")
    out = planner.plan_node(st(run_dir, human_in_the_loop=True))
    assert out["pipeline_halted"] is True and out["clarification_needed"] == "Which cache?"
    assert "search_notes" not in out          # a halt returns before any notes are produced


# ── ideation_node ────────────────────────────────────────────────────────────

def test_ideation_search_off(role, run_dir):
    role.result = FakePydantic()
    out = ideation.ideation_node(st(run_dir, use_search=False))
    kw = role.calls[0]
    assert kw["tools"] is None and kw["template_vars"]["search_hint"] == ""
    assert kw["template_vars"]["chat_block"] == "" and out["search_notes"] == ""


def test_ideation_search_on_returns_notes_for_the_plan_stage(role, run_dir):
    role.result = FakePydantic()
    role.history = [rec("idea q", "idea result")]
    out = ideation.ideation_node(st(run_dir, use_search=True))
    assert role.calls[0]["template_vars"]["search_hint"] == tools.SEARCH_HINT
    assert "idea result" in out["search_notes"]
    assert out["ideation_path"].endswith("ideation.json")


def test_ideation_records_truncation_escalation(role, run_dir):
    role.result = FakePydantic(_escalated_to="35b", _escalated_from="27b")
    out = ideation.ideation_node(st(run_dir))
    assert out["escalated_models"]["ideation"] == "35b"
    assert out["escalation_history"][0]["trigger"] == "truncation"


# ── drafter ──────────────────────────────────────────────────────────────────

def _draft_state(run_dir, **kw):
    kw.setdefault("iteration", 0)
    return st(run_dir, plan_spec=FakePydantic(), **kw)


@pytest.mark.parametrize("node", [drafter.draft_short_node, drafter.draft_node])
def test_draft_prompt_gets_search_results(role, run_dir, node):
    role.result = FakePydantic()
    node(_draft_state(run_dir, use_search=True, search_notes="Query: q\nFastAPI 0.115.2"))
    block = role.calls[0]["template_vars"]["correction_block"]
    assert "ENABLED" in block and "FastAPI 0.115.2" in block and "untrusted" in block


@pytest.mark.parametrize("node", [drafter.draft_short_node, drafter.draft_node])
def test_draft_prompt_says_search_is_enabled_even_with_no_results(role, run_dir, node):
    role.result = FakePydantic()
    node(_draft_state(run_dir, use_search=True, search_notes=""))
    block = role.calls[0]["template_vars"]["correction_block"]
    assert "ENABLED" in block and "No web search was run" in block


@pytest.mark.parametrize("node", [drafter.draft_short_node, drafter.draft_node])
def test_draft_prompt_is_untouched_when_search_is_off(role, run_dir, node):
    role.result = FakePydantic()
    node(_draft_state(run_dir, use_search=False, search_notes="leftover from an earlier turn"))
    assert role.calls[0]["template_vars"]["correction_block"] == ""


def test_draft_redraft_keeps_both_correction_and_search_context(role, run_dir):
    role.result = FakePydantic()
    verdict = type("V", (), {"description": "bad", "specific_issues": ["x"]})()
    drafter.draft_short_node(_draft_state(run_dir, use_search=True, search_notes="n", iteration=2,
                                          validation_verdict=verdict))
    block = role.calls[0]["template_vars"]["correction_block"]
    assert "[REDRAFT" in block and "ENABLED" in block


def test_draft_requires_a_plan(role, run_dir):
    with pytest.raises(ValueError, match="plan_spec missing"):
        drafter.draft_short_node(st(run_dir))
    with pytest.raises(ValueError, match="plan_spec missing"):
        drafter.draft_node(st(run_dir))


def test_draft_writes_artifact_and_guard_flags(role, run_dir):
    role.result = FakePydantic()
    out = drafter.draft_short_node(_draft_state(run_dir))
    assert out["draft_path"].endswith("draft.json")
    assert out["_guard_passed"] is True