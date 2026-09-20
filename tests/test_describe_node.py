"""nodes/describe.py — direct-answer node: search tool loop, request shape, runaway-thinking recovery."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from clients import llm
from clients import tools as tools_mod
from nodes import describe as D
from tests.fakes import Msg, completion, make_openai, tool_call

TASK = "search for today's date"


@pytest.fixture
def env(monkeypatch):
    e = SimpleNamespace(cfg={"base_url": "http://x/v1", "model_id": "q9", "name": "Qwen3.5-9B",
                             "temperature": 0.6, "top_p": 0.9},
                        budget=512, cap=16000, thinking_logs=[], stage_logs=[], searched=[])
    monkeypatch.setattr(D, "get_model_config", lambda k: e.cfg)
    monkeypatch.setattr(D, "load_prompt", lambda n: {"system": "You are a knowledgeable assistant."})
    monkeypatch.setattr(D, "resolve_role", lambda r: "9b")
    monkeypatch.setattr(D, "resolve_ultra_model", lambda r: "35b")
    monkeypatch.setattr(D, "_get_thinking_budget", lambda s: e.budget)
    monkeypatch.setattr(D, "_get_http_timeout", lambda: 30)
    monkeypatch.setattr(D, "_get_output_token_cap", lambda s: e.cap)
    monkeypatch.setattr(D, "_write_thinking_log", lambda rd, st, tb: e.thinking_logs.append(tb))
    monkeypatch.setattr(D, "_log_stage_entry", lambda *a, **k: e.stage_logs.append((a, k)))
    monkeypatch.setattr(D, "ensure_model_loaded", lambda k: False)
    monkeypatch.setattr(llm, "next_escalation_model", lambda stage, cur: None)
    monkeypatch.delenv("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE", raising=False)
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "search_web",
                        lambda a: e.searched.append(a["query"]) or "[Web search results]\n1. Friday, September 18, 2026")

    def go(script, state=None, task=TASK):
        fake = make_openai(list(script))
        e.fake = fake                                   # visible even if the node raises
        monkeypatch.setattr(D, "OpenAI", fake)
        rd = str(e.tmp)
        state = {} if state is None else state          # mutated in place by the node on truncation
        state.setdefault("run_dir", rd)
        state.setdefault("raw_text_input", task)
        return D.describe_node(state), fake, rd
    e.go = go
    return e


@pytest.fixture(autouse=True)
def _tmp(env, tmp_path):
    env.tmp = tmp_path


def answer_of(out):
    return json.load(open(out["final_output_path"]))["answer"]


# ── search off: behaviour is unchanged ───────────────────────────────────────

def test_search_off_single_call_no_tools_prompt_untouched(env):
    out, fake, _ = env.go([completion(Msg("I can't browse."))], {"use_search": False})
    c = fake.calls[0]
    assert len(fake.calls) == 1 and "tools" not in c and "tool_choice" not in c
    assert c["messages"][1]["content"] == TASK
    assert answer_of(out) == "I can't browse." and out["pipeline_complete"] is True
    assert out["search_notes"] == ""


def test_final_json_records_task_type(env):
    out, _, _ = env.go([completion(Msg("fine"))])
    assert json.load(open(out["final_output_path"]))["task_type"] == "describe"


# ── search on ────────────────────────────────────────────────────────────────

def test_search_round_trip(env):
    out, fake, _ = env.go([
        completion(Msg(tool_calls=[tool_call("call_1", "today's date")], reasoning="I should search."), finish="tool_calls", completion_tokens=20),
        completion(Msg("Today is Friday, September 18, 2026."), completion_tokens=15)], {"use_search": True})
    first, second = fake.calls
    assert first["tools"] and first["tool_choice"] == "auto"
    assert first["messages"][1]["content"].startswith("[Web search is ENABLED")
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert second["messages"][3]["tool_call_id"] == "call_1"
    assert env.searched == ["today's date"]
    assert answer_of(out) == "Today is Friday, September 18, 2026."
    assert "today's date" in out["search_notes"] and "September 18" in out["search_notes"]
    assert env.thinking_logs == ["I should search."]
    args, _ = env.stage_logs[0]
    assert args[5] == 35                       # tokens_out summed over both rounds


def test_search_on_but_model_answers_directly(env):
    out, _, _ = env.go([completion(Msg("Direct answer."))], {"use_search": True})
    assert out["search_notes"] == "" and answer_of(out) == "Direct answer."


def test_runaway_tool_calls_are_capped_then_forced_to_answer(env):
    script = [completion(Msg(tool_calls=[tool_call(f"c{i}", f"q{i}")]), finish="tool_calls") for i in range(1, 5)]
    out, fake, _ = env.go(script + [completion(Msg("Best effort."))], {"use_search": True})
    assert [c["tool_choice"] for c in fake.calls] == ["auto"] * 4 + ["none"]
    assert len(env.searched) == 4 and answer_of(out) == "Best effort."


def test_tool_exception_does_not_crash_the_node(env, monkeypatch):
    def boom(a):
        raise RuntimeError("searxng down")
    monkeypatch.setitem(tools_mod.TOOL_IMPLEMENTATIONS, "search_web", boom)
    out, fake, _ = env.go([completion(Msg(tool_calls=[tool_call("c1", "x")]), finish="tool_calls"),
                           completion(Msg("From memory."))], {"use_search": True})
    assert "Error running tool" in fake.calls[1]["messages"][3]["content"]
    assert out["pipeline_complete"] is True


def test_unknown_tool_and_bad_arguments_are_survivable(env):
    out, fake, _ = env.go([
        completion(Msg(tool_calls=[tool_call("c1", name="rm_rf"), tool_call("c2", raw_arguments="{oops")]), finish="tool_calls"),
        completion(Msg("ok"))], {"use_search": True})
    contents = [m["content"] for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert contents[0].startswith("Error: unknown tool") and out["pipeline_complete"]


# ── request shape: describe must send what every other node sends ────────────

def test_thinking_request_carries_reasoning_budget_and_top_p(env):
    _, fake, _ = env.go([completion(Msg("a"))])
    c = fake.calls[0]
    assert c["top_p"] == 0.9 and c["temperature"] == 0.6 and c["max_tokens"] == 16000
    assert c["extra_body"]["reasoning_budget"] == 512
    assert c["extra_body"]["thinking"] == {"type": "enabled", "budget_tokens": 512}


def test_top_p_defaults_to_0_95(env):
    env.cfg.pop("top_p")
    _, fake, _ = env.go([completion(Msg("a"))])
    assert fake.calls[0]["top_p"] == 0.95


def test_nowait_logit_bias_is_applied_when_configured(env):
    env.cfg["nowait_tokens"] = {"11": "Wait", "22": "Hmm"}
    _, fake, _ = env.go([completion(Msg("a"))])
    assert fake.calls[0]["extra_body"]["logit_bias"] == {"11": -100.0, "22": -100.0}


def test_no_logit_bias_when_not_configured(env):
    _, fake, _ = env.go([completion(Msg("a"))])
    assert "logit_bias" not in fake.calls[0]["extra_body"]


def test_zero_budget_disables_thinking(env):
    env.budget = 0
    _, fake, _ = env.go([completion(Msg("a"))])
    assert fake.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}


def test_output_cap_override_env_wins(env, monkeypatch):
    monkeypatch.setenv("PIPELINE_STEP_OUTPUT_CAP_OVERRIDE", "32000")
    _, fake, _ = env.go([completion(Msg("a"))])
    assert fake.calls[0]["max_tokens"] == 32000


# ── runaway-thinking recovery ────────────────────────────────────────────────

def test_cap_hit_mid_thought_retries_once_with_thinking_disabled(env):
    out, fake, _ = env.go([
        completion(Msg("", reasoning="Wait, let me reconsider... " * 50), finish="length", completion_tokens=16000),
        completion(Msg("Friday, September 18, 2026."), completion_tokens=12)])
    first, second = fake.calls
    assert first["extra_body"]["thinking"]["type"] == "enabled"
    assert first["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    # the retry must switch thinking off at BOTH levels, with no reasoning_budget
    assert second["extra_body"] == {"thinking": {"type": "disabled"},
                                    "chat_template_kwargs": {"enable_thinking": False}}
    assert answer_of(out) == "Friday, September 18, 2026." and out["pipeline_complete"] is True
    args, kw = env.stage_logs[0]
    assert args[5] == 16012                                               # tokens across BOTH attempts
    assert 0 < kw["think_ratio"] < 1


def test_unterminated_think_tag_in_content_also_triggers_recovery(env):
    out, fake, _ = env.go([
        completion(Msg("<think>Wait, hmm, wait..."), finish="length"),
        completion(Msg("Answer."))])
    assert len(fake.calls) == 2 and answer_of(out) == "Answer."


def test_recovery_keeps_the_logit_bias(env):
    env.cfg["nowait_tokens"] = {"11": "Wait"}
    _, fake, _ = env.go([completion(Msg(""), finish="length"), completion(Msg("ok"))])
    assert fake.calls[1]["extra_body"]["logit_bias"] == {"11": -100.0}


def test_truncation_with_partial_answer_raises_without_a_retry(env):
    """A cut-off ANSWER (not a thinking loop) is a genuinely long answer: raise, don't silently re-roll."""
    with pytest.raises(llm.TruncatedOutputError) as exc:
        env.go([completion(Msg("a long answer that got cut o"), finish="length")])
    assert exc.value.stage == "describe"
    assert len(env.fake.calls) == 1


def test_truncation_with_thinking_disabled_does_not_retry(env):
    env.budget = 0
    with pytest.raises(llm.TruncatedOutputError):
        env.go([completion(Msg(""), finish="length")])
    assert len(env.fake.calls) == 1                       # nothing to switch off, so no second attempt


def test_recovery_that_also_truncates_raises(env):
    with pytest.raises(llm.TruncatedOutputError):
        env.go([completion(Msg(""), finish="length"), completion(Msg("cut off aga"), finish="length")])
    assert len(env.fake.calls) == 2                       # exactly one retry, never a loop


def test_truncation_escalates_model_for_the_retry_wrapper(env, monkeypatch):
    monkeypatch.setattr(llm, "next_escalation_model", lambda stage, cur: "35b")
    state = {}
    with pytest.raises(llm.TruncatedOutputError):
        env.go([completion(Msg("partial"), finish="length")], state)
    assert state["escalated_models"] == {"describe": "35b"}
    assert state["escalation_history"][0]["trigger"] == "truncation"


# ── model selection ──────────────────────────────────────────────────────────

def test_escalated_model_override_wins(env):
    assert D._resolve_describe_model({"escalated_models": {"describe": "35b"}}) == "35b"


def test_ultra_profile_uses_ladder_final_entry(env):
    assert D._resolve_describe_model({"profile": "ultra"}) == "35b"


def test_default_uses_role_mapping(env):
    assert D._resolve_describe_model({}) == "9b"


# ── thinking that arrives inside the content (<think> tags) ──────────────────

def test_in_content_think_block_is_logged_and_stripped_from_the_answer(env):
    out, _, _ = env.go([completion(Msg("<think>plan the reply</think>The answer."))])
    assert answer_of(out) == "The answer."
    assert env.thinking_logs == ["plan the reply"]


def test_runaway_thinking_is_logged_before_the_retry(env):
    env.go([completion(Msg("<think>Wait, hmm, wait..."), finish="length"), completion(Msg("Answer."))])
    assert env.thinking_logs[0] == "Wait, hmm, wait..."


def test_think_ratio_is_zero_without_any_thinking(env):
    env.go([completion(Msg("plain answer"))])
    assert env.stage_logs[0][1]["think_ratio"] == 0.0


# ── resilience + model selection details ─────────────────────────────────────

def test_model_manager_failure_is_not_fatal(env, monkeypatch, caplog):
    def broken(key):
        raise ConnectionError("manager unreachable")
    monkeypatch.setattr(D, "ensure_model_loaded", broken)
    out, _, _ = env.go([completion(Msg("still answers"))])
    assert answer_of(out) == "still answers" and "assuming 9b already running" in caplog.text


def test_profile_role_override_comes_from_routing_config(env, monkeypatch):
    import yaml
    monkeypatch.setattr(yaml, "safe_load", lambda f: {"pipeline_profiles": {"long": {"role_overrides": {"describe": "35b"}}}})
    assert D._resolve_describe_model({"profile": "long"}) == "35b"


def test_profile_without_an_override_uses_the_role_default(env, monkeypatch):
    import yaml
    monkeypatch.setattr(yaml, "safe_load", lambda f: {"pipeline_profiles": {"short": {"role_overrides": {}}}})
    assert D._resolve_describe_model({"profile": "short"}) == "9b"


def test_unreadable_routing_config_falls_back_to_the_role_default(env, monkeypatch, caplog):
    import yaml

    def broken(f):
        raise ValueError("bad yaml")
    monkeypatch.setattr(yaml, "safe_load", broken)
    assert D._resolve_describe_model({"profile": "long"}) == "9b"
    assert "Could not read pipeline_profiles" in caplog.text


def test_requested_profile_is_used_when_no_resolved_profile_yet(env, monkeypatch):
    import yaml
    monkeypatch.setattr(yaml, "safe_load", lambda f: {"pipeline_profiles": {"medium": {"role_overrides": {"describe": "27b"}}}})
    assert D._resolve_describe_model({"requested_profile": "medium"}) == "27b"


def test_existing_escalation_state_is_passed_through(env):
    state = {"escalated_models": {"describe": "35b"}, "escalation_history": [{"stage": "describe"}]}
    out, fake, _ = env.go([completion(Msg("ok"))], state)
    assert out["escalated_models"] == {"describe": "35b"}
    assert out["escalation_history"] == [{"stage": "describe"}]