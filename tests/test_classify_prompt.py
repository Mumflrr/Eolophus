"""config/prompts/classify.yaml + schemas/task_classification.py — offline checks (no model needed)."""
from __future__ import annotations

import re

import pytest
import yaml
from pydantic import ValidationError

from clients import llm
from schemas.task_classification import Complexity, Mode, TaskClassification, TaskType


@pytest.fixture(scope="module")
def prompt(project_root):
    return yaml.safe_load((project_root / "config" / "prompts" / "classify.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def system(prompt):
    return prompt["system"]


@pytest.fixture(scope="module")
def examples(system):
    """(mode, type, complexity, confidence) for every `→ mode=..., type=...` example in the prompt."""
    block = system.split("EXAMPLES:")[1]
    return re.findall(r"→ mode=(\w+), type=(\w+), complexity=(\w+), confidence=(\w+)", block)


# ── file hygiene ─────────────────────────────────────────────────────────────

def test_prompt_has_the_keys_load_prompt_expects(prompt):
    assert prompt["role"] == "classify" and prompt["thinking"] is False
    assert {"system", "user_template"} <= set(prompt)


def test_system_prompt_has_no_braces(system):
    """_safe_format runs over the system text — a stray {word} would be treated as a placeholder."""
    assert "{" not in system and "}" not in system


def test_user_template_has_exactly_the_task_placeholder(prompt):
    assert re.findall(r"\{(\w+)\}", prompt["user_template"]) == ["task"]


def test_renders_through_the_real_prompt_builder():
    msgs = llm.build_messages_from_prompt("classify", {"task": "search for today's date"}, TaskClassification)
    user = next(m["content"] for m in msgs if m["role"] == "user")
    assert "search for today's date" in user and "{task}" not in user


# ── the prompt must not teach the model something the schema rejects ─────────

def test_every_example_is_a_legal_classification(examples):
    assert len(examples) >= 10
    for mode, typ, cx, conf in examples:
        kw = dict(mode=mode, task_type=typ, complexity=cx, decompose=False, reasoning="example", confidence=conf)
        if conf == "low":
            kw["clarification_question"] = "q?"
        TaskClassification(**kw)


def test_prompt_does_not_offer_modes_the_schema_rejects(system):
    modes_block = system.split("Complexity:")[0]
    listed = set(re.findall(r"^\s+(\w+)\s+[—-]", modes_block, re.M))
    assert listed == {m.value for m in Mode}, "prompt advertises modes the Mode enum doesn't have"


@pytest.mark.parametrize("bad", ["medium", "ultra"])
def test_schema_rejects_the_tiers_the_old_prompt_advertised(bad):
    with pytest.raises(ValidationError):
        TaskClassification(mode=bad, task_type="coding", complexity="simple", decompose=False,
                           reasoning="x", confidence="high")


def test_every_task_type_and_complexity_is_defined_in_the_prompt(system):
    for t in TaskType:
        assert re.search(rf"^\s+{t.value}\s+—", system, re.M), f"task type {t.value!r} not defined"
    for c in Complexity:
        assert re.search(rf"^\s+{c.value}\s+—", system, re.M), f"complexity {c.value!r} not defined"


def test_examples_cover_every_task_type(examples):
    assert {t for _, t, _, _ in examples} == {t.value for t in TaskType}


# ── the search guidance the prompt is supposed to carry ───────────────────────

def test_search_is_described_as_a_means_not_a_task_type(system):
    assert "MEANS, never a task type" in system
    assert "build verb" in system


@pytest.mark.parametrize("phrase,expected_type", [
    ("search for today's date", "describe"),
    ("can you search the internet?", "describe"),
    ("compare the current top Python web frameworks", "describe"),
    ("queries the SearXNG JSON API", "coding"),
    ("look up the latest stable pydantic version and write", "coding"),
    ("adding web search to a local LLM pipeline", "ideation"),
])
def test_prompt_has_a_worked_example_for_each_side_of_the_search_boundary(system, phrase, expected_type):
    block = system.split("EXAMPLES:")[1]
    i = block.index(phrase)
    following = re.search(r"type=(\w+)", block[i:]).group(1)
    assert following == expected_type


# ── schema rules the prompt tells the model about ────────────────────────────

def test_nulls_are_only_legal_at_low_confidence():
    base = dict(mode=None, task_type=None, complexity=None, decompose=False, reasoning="x")
    TaskClassification(**base, confidence="low", clarification_question="q?")
    for conf in ("high", "medium"):
        with pytest.raises(ValidationError):
            TaskClassification(**base, confidence=conf)


def test_partial_nulls_are_rejected_at_high_confidence():
    with pytest.raises(ValidationError):
        TaskClassification(mode="short", task_type=None, complexity="simple", decompose=False,
                           reasoning="x", confidence="high")


def test_prompt_asks_for_a_short_reasoning_field(system):
    """A classification is ~150 tokens; the reasoning field is where a model can ramble."""
    assert "ONE short sentence" in system and "do not deliberate at length" in system