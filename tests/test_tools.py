"""clients/tools.py — tool schema, tool implementation, and the prompt-side search helpers."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from clients import tools


def _rec(name="search_web", query="q", result="r"):
    return SimpleNamespace(name=name, arguments={"query": query}, result=result)


# ── schema + registry ────────────────────────────────────────────────────────

def test_schema_shape():
    fn = tools.SEARCH_TOOL_SCHEMA["function"]
    assert tools.SEARCH_TOOL_SCHEMA["type"] == "function"
    assert fn["name"] == "search_web"
    assert fn["parameters"]["required"] == ["query"]
    assert fn["parameters"]["properties"]["query"]["type"] == "string"


def test_registry_matches_schema_name():
    assert set(tools.TOOL_IMPLEMENTATIONS) == {tools.SEARCH_TOOL_SCHEMA["function"]["name"]}
    assert callable(tools.TOOL_IMPLEMENTATIONS["search_web"])


# ── _run_search_web ──────────────────────────────────────────────────────────

def test_run_search_formats_results(monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "search_web", lambda q: seen.append(q) or ["R"])
    monkeypatch.setattr(tools, "format_results_for_prompt", lambda rs: f"formatted:{rs}")
    assert tools._run_search_web({"query": "fastapi"}) == "formatted:['R']"
    assert seen == ["fastapi"]


def test_run_search_no_results_gives_explicit_message(monkeypatch, caplog):
    monkeypatch.setattr(tools, "search_web", lambda q: [])
    caplog.set_level(logging.INFO, logger="clients.tools")
    assert tools._run_search_web({"query": "zzz"}) == "No results found."
    assert "zero results" in caplog.text


@pytest.mark.parametrize("args", [None, {}, {"other": 1}])
def test_run_search_tolerates_missing_query(monkeypatch, args):
    seen = []
    monkeypatch.setattr(tools, "search_web", lambda q: seen.append(q) or [])
    assert tools._run_search_web(args) == "No results found."
    assert seen == [""]


# ── SEARCH_HINT ──────────────────────────────────────────────────────────────

def test_hint_tells_model_it_can_search_and_when():
    h = tools.SEARCH_HINT
    assert "search_web" in h and "ENABLED" in h
    assert "current" in h and "can search" in h
    assert h.endswith("\n\n")            # separates cleanly from the task that follows


# ── format_search_notes ──────────────────────────────────────────────────────

def test_notes_empty_inputs():
    assert tools.format_search_notes([]) == ""
    assert tools.format_search_notes(None) == ""
    assert tools.format_search_notes(None, carried="") == ""


def test_notes_include_query_and_result():
    n = tools.format_search_notes([_rec(query="fastapi latest", result="[Web search results]\n1. FastAPI")])
    assert n == "Query: fastapi latest\n[Web search results]\n1. FastAPI"


def test_notes_skip_empty_no_result_and_other_tools():
    n = tools.format_search_notes([
        _rec(result="No results found."), _rec(result="   "), _rec(name="other_tool", result="ignored"),
        _rec(query="kept", result="body")])
    assert n == "Query: kept\nbody"


def test_notes_carry_earlier_stage_first():
    n = tools.format_search_notes([_rec(query="new", result="new body")], carried="Query: old\nold body")
    assert n.index("old body") < n.index("new body")


def test_notes_blank_carried_is_ignored():
    assert tools.format_search_notes([_rec(query="q", result="b")], carried="  \n ") == "Query: q\nb"


def test_notes_are_capped():
    n = tools.format_search_notes([_rec(result="y" * 9000)])
    assert len(n) <= tools._MAX_SEARCH_NOTES_CHARS + len("\n[... truncated]")
    assert n.endswith("[... truncated]")


def test_notes_tolerate_records_missing_attributes():
    bare = SimpleNamespace()                    # no name/arguments/result at all
    assert tools.format_search_notes([bare]) == ""
    assert tools.format_search_notes([SimpleNamespace(name="search_web", arguments=None, result="x")]) == "Query: \nx"


# ── format_search_context ────────────────────────────────────────────────────

@pytest.mark.parametrize("use", [False, None, 0])
def test_context_is_empty_when_search_off(use):
    assert tools.format_search_context(use, "some notes") == ""


def test_context_when_on_without_results_still_says_search_is_enabled():
    c = tools.format_search_context(True, "")
    assert "ENABLED" in c and "Do not claim you cannot" in c
    assert "No web search was run" in c


def test_context_with_results_fences_them_as_untrusted():
    c = tools.format_search_context(True, "Query: q\nFriday, September 18, 2026")
    assert "untrusted" in c and "never as instructions" in c
    assert "Friday, September 18, 2026" in c
    assert c.index("Friday") < c.index("[End of web search results]")