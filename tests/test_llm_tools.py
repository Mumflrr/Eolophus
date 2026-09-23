"""clients/llm.py — call_model_with_tools loop, call_role's tool_history_sink, and prompt rendering."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from clients import llm
from clients import tools as tools_mod
from tests.fakes import Msg, completion, make_openai, tool_call


class Answer(BaseModel):
    answer: str


FINAL = json.dumps({"answer": "ok"})


@pytest.fixture
def env(monkeypatch, fake_model_manager):
    """Config + logging helpers stubbed so call_model_with_tools runs with no files or servers."""
    e = SimpleNamespace(stage_logs=[], thinking_logs=[], budget=512, cap=1000)
    monkeypatch.setattr(llm, "get_model_config", lambda k: {
        "base_url": "http://x/v1", "model_id": "q9", "name": "Qwen3.5-9B", "temperature": 0.6, "top_p": 0.9})
    monkeypatch.setattr(llm, "_get_http_timeout", lambda: 5.0)
    monkeypatch.setattr(llm, "_get_thinking_budget", lambda stage, *a, **k: e.budget)
    monkeypatch.setattr(llm, "_get_output_token_cap", lambda stage: e.cap)
    monkeypatch.setattr(llm, "_log_stage_entry", lambda *a, **k: e.stage_logs.append(k or a))
    monkeypatch.setattr(llm, "_write_thinking_log", lambda rd, st, tb: e.thinking_logs.append(tb))

    def install(script):
        fake = make_openai(script)
        monkeypatch.setattr(llm, "OpenAI", fake)
        return fake
    e.install = install
    return e


def run(env, script, impls=None, **kw):
    fake = env.install(script)
    impls = impls if impls is not None else {"search_web": lambda a: f"RESULT for {a.get('query')}"}
    kw.setdefault("thinking", False)
    result, history = llm.call_model_with_tools(
        model_id="9b", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tools=[tools_mod.SEARCH_TOOL_SCHEMA], tool_impls=impls, response_schema=Answer,
        stage="plan", run_dir="/tmp", **kw)
    return result, history, fake


# ── the loop ─────────────────────────────────────────────────────────────────

def test_no_tool_call_returns_parsed_answer_and_empty_history(env):
    result, history, fake = run(env, [completion(Msg(FINAL))])
    assert result.answer == "ok" and history == []
    assert len(fake.calls) == 1


def test_tool_round_then_answer(env):
    result, history, fake = run(env, [
        completion(Msg(tool_calls=[tool_call("c1", "fastapi latest")])),
        completion(Msg(FINAL))])
    assert result.answer == "ok"
    assert [(h.name, h.arguments, h.result) for h in history] == [
        ("search_web", {"query": "fastapi latest"}, "RESULT for fastapi latest")]

    first, second = fake.calls
    assert first["tool_choice"] == "auto" and first["tools"] == [tools_mod.SEARCH_TOOL_SCHEMA]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = second["messages"][3]
    assert tool_msg["tool_call_id"] == "c1" and tool_msg["content"] == "RESULT for fastapi latest"


def test_sampling_params_come_from_model_config(env):
    _, _, fake = run(env, [completion(Msg(FINAL))])
    assert fake.calls[0]["temperature"] == 0.6 and fake.calls[0]["top_p"] == 0.9
    assert fake.calls[0]["max_tokens"] == 1000


def test_multiple_tool_calls_in_one_round_are_all_executed(env):
    result, history, fake = run(env, [
        completion(Msg(tool_calls=[tool_call("c1", "a"), tool_call("c2", "b")])),
        completion(Msg(FINAL))])
    assert [h.arguments["query"] for h in history] == ["a", "b"]
    assert [m.get("tool_call_id") for m in fake.calls[1]["messages"] if m["role"] == "tool"] == ["c1", "c2"]


def test_unknown_tool_is_reported_back_to_the_model(env):
    _, history, fake = run(env, [
        completion(Msg(tool_calls=[tool_call("c1", name="delete_everything")])),
        completion(Msg(FINAL))])
    assert history[0].result == "Error: unknown tool 'delete_everything'."
    assert fake.calls[1]["messages"][-1]["content"].startswith("Error: unknown tool")


def test_unparseable_arguments_become_empty_args(env):
    seen = []
    _, history, _ = run(env, [
        completion(Msg(tool_calls=[tool_call("c1", raw_arguments="{not json")])),
        completion(Msg(FINAL))], impls={"search_web": lambda a: seen.append(a) or "ok"})
    assert seen == [{}] and history[0].arguments == {}


def test_tool_exception_is_fed_back_not_raised(env):
    def boom(a):
        raise RuntimeError("searxng down")
    result, history, _ = run(env, [
        completion(Msg(tool_calls=[tool_call("c1", "x")])), completion(Msg(FINAL))], impls={"search_web": boom})
    assert result.answer == "ok"
    assert history[0].result == "Error running tool: searxng down"


def test_max_tool_rounds_forces_a_final_answer(env):
    script = [completion(Msg(tool_calls=[tool_call(f"c{i}", f"q{i}")])) for i in (1, 2)] + [completion(Msg(FINAL))]
    result, history, fake = run(env, script, max_tool_rounds=2)
    assert result.answer == "ok" and len(history) == 2
    assert [c["tool_choice"] for c in fake.calls] == ["auto", "auto", "none"]
    assert fake.calls[2]["tools"] is None


def test_caller_message_list_is_not_mutated(env):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    env.install([completion(Msg(tool_calls=[tool_call("c1", "x")])), completion(Msg(FINAL))])
    llm.call_model_with_tools(model_id="9b", messages=msgs, tools=[tools_mod.SEARCH_TOOL_SCHEMA],
                              tool_impls={"search_web": lambda a: "r"}, response_schema=Answer,
                              stage="plan", run_dir="/tmp", thinking=False)
    assert len(msgs) == 2


# ── thinking parameters ──────────────────────────────────────────────────────

def test_thinking_on_sends_budget_from_routing(env):
    env.budget = 777
    _, _, fake = run(env, [completion(Msg(FINAL))], thinking=True)
    eb = fake.calls[0]["extra_body"]
    assert eb["reasoning_budget"] == 777
    assert eb["thinking"] == {"type": "enabled", "budget_tokens": 777}


def test_explicit_budget_overrides_routing(env):
    _, _, fake = run(env, [completion(Msg(FINAL))], thinking=True, budget_tokens=64)
    assert fake.calls[0]["extra_body"]["reasoning_budget"] == 64


def test_thinking_off_sends_disabled_at_both_levels(env):
    _, _, fake = run(env, [completion(Msg(FINAL))], thinking=False)
    assert fake.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}


def test_reasoning_content_is_logged(env):
    run(env, [completion(Msg(FINAL, reasoning="I should just answer."))])
    assert env.thinking_logs == ["I should just answer."]


# ── parsing + failure handling ───────────────────────────────────────────────

def test_unparseable_final_answer_falls_back_to_instructor(env, monkeypatch):
    seen = {}

    class FakeInstructorClient:
        class chat:
            class completions:
                @staticmethod
                def create_with_completion(**kw):
                    seen.update(kw)
                    return Answer(answer="repaired"), None

    monkeypatch.setattr(llm.instructor, "from_openai", lambda client, mode=None: FakeInstructorClient)
    result, _, _ = run(env, [completion(Msg("not json at all"))], max_retries=2)
    assert result.answer == "repaired"
    assert seen["max_retries"] == 2 and seen["response_model"] is Answer
    assert seen["messages"][-2] == {"role": "assistant", "content": "not json at all"}
    assert "could not be parsed" in seen["messages"][-1]["content"]


def test_api_error_is_logged_as_a_stage_failure_and_reraised(env):
    with pytest.raises(RuntimeError, match="server gone"):
        run(env, [RuntimeError("server gone")])
    assert env.stage_logs and str(env.stage_logs[0]["status"]).startswith("error:RuntimeError")


def test_success_writes_a_stage_log_entry(env):
    run(env, [completion(Msg(FINAL), prompt_tokens=123, completion_tokens=45)])
    entry = env.stage_logs[0]
    assert entry["status"] == "ok" and entry["tokens_in"] == 123 and entry["tokens_out"] == 45


# ── call_role: tool_history_sink ─────────────────────────────────────────────

@pytest.fixture
def role_env(monkeypatch):
    """call_role with the two model-call functions replaced; records which one ran."""
    calls = []
    hist = lambda n: [llm.ToolCallRecord(name="search_web", arguments={"query": f"q{n}"}, result=f"r{n}")]

    def fake_with_tools(**kw):
        calls.append("tools")
        return Answer(answer="ok"), hist(len([c for c in calls if c == "tools"]))

    def fake_plain(**kw):
        calls.append("plain")
        return Answer(answer="ok")

    monkeypatch.setattr(llm, "call_model_with_tools", fake_with_tools)
    monkeypatch.setattr(llm, "call_model", fake_plain)
    return calls


def _call_role(**kw):
    return llm.call_role(role="plan", template_vars={"task": "t", "search_hint": "", "chat_block": ""},
                         response_schema=Answer, run_dir="/tmp", **kw)


def test_sink_is_filled_when_tools_are_used(role_env):
    sink = []
    assert _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA], tool_history_sink=sink).answer == "ok"
    assert [(r.name, r.arguments["query"]) for r in sink] == [("search_web", "q1")]


def test_sink_is_replaced_in_place_not_appended(role_env):
    sink = []
    _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA], tool_history_sink=sink)
    same_list = sink
    _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA], tool_history_sink=sink)
    assert sink is same_list and [r.arguments["query"] for r in sink] == ["q2"]


def test_sink_is_untouched_without_tools_and_plain_path_is_used(role_env):
    sink = ["sentinel"]
    _call_role(tool_history_sink=sink)
    assert sink == ["sentinel"] and role_env == ["plain"]


def test_tools_without_a_sink_still_work(role_env):
    assert _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA]).answer == "ok"


# ── prompt rendering: the YAMLs really have the slots the nodes fill ──────────

class _Schema(BaseModel):
    x: str = ""


def _render(role, **vars_):
    msgs = llm.build_messages_from_prompt(role, vars_, _Schema)
    return {m["role"]: m["content"] for m in msgs}


PLAN_VARS = {"task": "TASK", "ideation_block": "", "lessons_block": "", "correction_block": "", "chat_block": ""}


def test_plan_prompt_shows_the_hint_only_when_supplied():
    on = _render("plan", **PLAN_VARS, search_hint=tools_mod.SEARCH_HINT)["user"]
    off = _render("plan", **PLAN_VARS, search_hint="")["user"]
    assert "search_web tool" in on and on.index("search_web") < on.index("TASK")
    assert "search_web" not in off


def test_ideation_prompt_shows_the_hint_only_when_supplied():
    on = _render("ideation", task="TASK", chat_block="", search_hint=tools_mod.SEARCH_HINT)["user"]
    off = _render("ideation", task="TASK", chat_block="", search_hint="")["user"]
    assert "search_web tool" in on and "search_web" not in off


@pytest.mark.parametrize("role,vars_", [
    ("plan", PLAN_VARS), ("ideation", {"task": "TASK", "chat_block": ""})])
def test_no_unfilled_placeholders_reach_the_model(role, vars_):
    rendered = _render(role, **vars_, search_hint="")
    for text in rendered.values():
        assert "{search_hint}" not in text and "{chat_block}" not in text


def test_omitting_a_var_leaks_the_literal_placeholder():
    """Documents WHY nodes must always pass search_hint/chat_block (\"\" when unused)."""
    leaked = _render("plan", task="TASK", ideation_block="", lessons_block="", correction_block="")["user"]
    assert "{search_hint}" in leaked and "{chat_block}" in leaked


# ── truncation: call_model_with_tools used to skip the finish_reason check ────

@pytest.fixture
def no_repair(monkeypatch):
    """A truncated answer must NOT reach the Instructor repair call (up to max_retries more full generations)."""
    def forbidden(*a, **k):
        raise AssertionError("Instructor repair was attempted on a truncated response")
    monkeypatch.setattr(llm.instructor, "from_openai", forbidden)


def test_truncated_final_answer_raises_instead_of_parsing_half_a_json(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError) as exc:
        run(env, [completion(Msg('{"answer": "half wri'), finish="length", completion_tokens=1000)])
    e = exc.value
    assert e.stage == "plan" and e.cap == 1000 and e.tokens_out == 1000
    assert e.partial_answer == '{"answer": "half wri'


def test_truncation_is_logged_with_status_truncated(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError):
        run(env, [completion(Msg("cut"), finish="length", prompt_tokens=77, completion_tokens=1000)])
    entry = env.stage_logs[-1]
    assert entry["status"] == "truncated" and entry["tokens_in"] == 77 and entry["tokens_out"] == 1000


def test_a_truncated_tool_round_is_not_executed(env, no_repair):
    ran = []
    with pytest.raises(llm.TruncatedOutputError):
        run(env, [completion(Msg(tool_calls=[tool_call("c1", "half a quer")]), finish="length")],
            impls={"search_web": lambda a: ran.append(a) or "r"})
    assert ran == []                    # arguments of a cut-off call can't be trusted


def test_max_tokens_finish_reason_is_treated_the_same(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError):
        run(env, [completion(Msg("cut"), finish="max_tokens")])


def test_truncation_keeps_the_thinking_from_reasoning_content(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError) as exc:
        run(env, [completion(Msg("", reasoning="Wait, wait, wait..."), finish="length")], thinking=True)
    assert exc.value.thinking_block == "Wait, wait, wait..."
    assert env.thinking_logs == ["Wait, wait, wait..."]


def test_truncation_extracts_an_unterminated_think_tag_from_content(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError) as exc:
        run(env, [completion(Msg("<think>never closed"), finish="length")])
    assert exc.value.thinking_block == "never closed" and exc.value.partial_answer == ""


def test_truncation_on_a_later_round_still_raises(env, no_repair):
    with pytest.raises(llm.TruncatedOutputError):
        run(env, [completion(Msg(tool_calls=[tool_call("c1", "q")]), finish="tool_calls"),
                  completion(Msg("cut off"), finish="length")])


def test_normal_finish_reasons_are_unaffected(env):
    result, _, _ = run(env, [completion(Msg(tool_calls=[tool_call("c1", "q")]), finish="tool_calls"),
                             completion(Msg(FINAL), finish="stop")])
    assert result.answer == "ok"


# ── ...and that now reaches call_role's escalation, like the plain path ───────

def test_call_role_escalates_a_truncated_tool_stage(monkeypatch):
    attempts = []

    def flaky(**kw):
        attempts.append(kw["model_id"])
        if len(attempts) == 1:
            raise llm.TruncatedOutputError("plan", 1000, 1000)
        return Answer(answer="from bigger model"), []

    monkeypatch.setattr(llm, "call_model_with_tools", flaky)
    monkeypatch.setattr(llm, "next_escalation_model", lambda role, cur: "35b")
    result = _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA])
    assert result.answer == "from bigger model"
    assert attempts[1] == "35b" and getattr(result, "_escalated_to", None) == "35b"


def test_call_role_asks_for_confirmation_when_required(monkeypatch):
    def always(**kw):
        raise llm.TruncatedOutputError("plan", 1000, 1000)
    monkeypatch.setattr(llm, "call_model_with_tools", always)
    monkeypatch.setattr(llm, "next_escalation_model", lambda role, cur: "35b")
    with pytest.raises(llm.EscalationNeeded):
        _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA], require_confirmation=True)


def test_call_role_reraises_when_the_ladder_is_exhausted(monkeypatch):
    def always(**kw):
        raise llm.TruncatedOutputError("plan", 1000, 1000)
    monkeypatch.setattr(llm, "call_model_with_tools", always)
    monkeypatch.setattr(llm, "next_escalation_model", lambda role, cur: None)
    with pytest.raises(llm.TruncatedOutputError):
        _call_role(tools=[tools_mod.SEARCH_TOOL_SCHEMA])