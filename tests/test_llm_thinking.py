"""
clients/llm.py — how thinking is requested and observed:
  _get_thinking_budget, _build_thinking_extra_body, _stream_completion's reasoning capture,
  and call_model's use of the routing.yaml budget.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from clients import llm


# ── _get_thinking_budget ─────────────────────────────────────────────────────

@pytest.fixture
def routing(monkeypatch):
    """
    Make the budget / switch lookups see a routing config we control instead of config/routing.yaml.

    config/loader.py caches the parsed YAML, so patching yaml.safe_load alone only works for whichever test
    happens to run while the cache is still empty. Two cases, because they exercise different layers:

      routing({...})            a config that PARSED fine. Replace the cached-config accessor, so the
                                readers (get_thinking_budget / get_thinking_control_flag) run for real
                                against our dict. A sentinel probe fails loudly if that patch can't reach
                                them (e.g. the loader starts binding the accessor at import time).

      routing(SomeException)    a config that COULD NOT BE READ. The "fall back to the default" behaviour
                                lives inside get_routing_config itself (it catches the read/parse error),
                                and the readers deliberately call it with no try/except of their own. So
                                replacing the accessor with a raiser would test nothing real: it skips the
                                very code under test and hands the readers an exception they were never
                                meant to see. Instead make the FILE READ fail, one layer below, and
                                reload the loader so no cache filled by an earlier test can mask it.
                                The reload re-runs the module in the same module dict, so the aliases
                                clients/llm.py holds (_get_thinking_budget etc.) see the fresh, empty cache.

    Teardown ORDER matters for the exception case, which is why yaml.safe_load is patched by hand here and
    not through monkeypatch: the loader must be reloaded AFTER the real safe_load is back. With
    monkeypatch, its undo runs after this fixture's teardown, so a reload in teardown re-reads through the
    still-broken safe_load, re-caches the failure, and the NEXT test silently gets default values.
    """
    import importlib
    import yaml
    from config import loader

    real_safe_load = yaml.safe_load
    patched_yaml = {"on": False}

    def install(data):
        if isinstance(data, Exception):
            def unreadable(*a, **k):
                raise data
            yaml.safe_load = unreadable
            patched_yaml["on"] = True
            importlib.reload(loader)
            return

        def accessor(*a, **k):
            return data
        monkeypatch.setattr(loader, "get_routing_config", accessor)

        probe = "__routing_fixture_probe__"
        probed = dict(data, thinking_budgets={**(data.get("thinking_budgets") or {}), probe: 31337})
        monkeypatch.setattr(loader, "get_routing_config", lambda *a, **k: probed)
        assert llm._get_thinking_budget(probe) == 31337, (
            "routing fixture can't reach the loader: config.loader.get_thinking_budget no longer reads "
            "get_routing_config by name at call time, so this fixture must patch wherever it binds it")
        monkeypatch.setattr(loader, "get_routing_config", accessor)

    try:
        yield install
    finally:
        if patched_yaml["on"]:
            yaml.safe_load = real_safe_load     # undo FIRST ...
            importlib.reload(loader)            # ... then drop the cache the failure populated


def test_a_configured_zero_reads_back_as_zero_not_2048(routing):
    """Was `int(x) if x else 2048` — falsy for 0, so every 'NO THINKING' stage claimed 2048."""
    routing({"thinking_budgets": {"bugfix": 0}})
    assert llm._get_thinking_budget("bugfix") == 0


def test_positive_and_unlimited_budgets(routing):
    routing({"thinking_budgets": {"plan": 1024, "ultra_plan": -1}})
    assert llm._get_thinking_budget("plan") == 1024
    assert llm._get_thinking_budget("ultra_plan") == -1


def test_unlisted_or_null_stage_defaults_to_2048(routing):
    routing({"thinking_budgets": {"explicit_null": None}})
    assert llm._get_thinking_budget("nope") == 2048
    assert llm._get_thinking_budget("explicit_null") == 2048


def test_complexity_nested_budgets(routing):
    routing({"thinking_budgets": {"draft": {"simple": 100, "moderate": 200}}})
    assert llm._get_thinking_budget("draft", "simple") == 100
    assert llm._get_thinking_budget("draft") == 200
    assert llm._get_thinking_budget("draft", "complex") == 200      # falls back to moderate


def test_unreadable_config_defaults_to_2048(routing):
    routing(OSError("no file"))
    assert llm._get_thinking_budget("plan") == 2048


def test_real_routing_yaml_budgets():
    assert llm._get_thinking_budget("bugfix") == 0
    assert llm._get_thinking_budget("classify") == 0
    assert llm._get_thinking_budget("plan") == 1024
    assert llm._get_thinking_budget("describe") == 512


# ── the chat_template_kwargs switch ──────────────────────────────────────────

def test_switch_reads_routing_yaml(routing):
    routing({"thinking_control": {"chat_template_kwargs": False}})
    assert llm._thinking_control_flag("chat_template_kwargs", True) is False
    routing({"thinking_control": {"chat_template_kwargs": True}})
    assert llm._thinking_control_flag("chat_template_kwargs", False) is True


@pytest.mark.parametrize("cfg", [{}, {"thinking_control": None}, {"thinking_control": {}}])
def test_switch_default_applies_when_absent(routing, cfg):
    routing(cfg)
    assert llm._thinking_control_flag("chat_template_kwargs", True) is True
    assert llm._thinking_control_flag("chat_template_kwargs", False) is False


def test_switch_default_applies_when_config_is_unreadable(routing):
    routing(OSError("no file"))
    assert llm._thinking_control_flag("chat_template_kwargs", True) is True


def test_real_routing_yaml_turns_the_switch_on():
    assert llm._thinking_control_flag("chat_template_kwargs", False) is True, \
        "routing.yaml needs `thinking_control: {chat_template_kwargs: true}`"


# ── _build_thinking_extra_body ───────────────────────────────────────────────

@pytest.fixture
def switch(monkeypatch):
    def set_(on):
        monkeypatch.setattr(llm, "_thinking_control_flag", lambda name, default: on)
    return set_


CASES = [
    # use_thinking, budget, expected thinking-part, enable_thinking
    (False, 512,  {"thinking": {"type": "disabled"}}, False),
    (False, None, {"thinking": {"type": "disabled"}}, False),
    (True,  0,    {"thinking": {"type": "disabled"}}, False),           # routing 0 == "NO THINKING"
    (True,  None, {"thinking": {"type": "enabled"}}, True),             # unlimited
    (True,  -1,   {"thinking": {"type": "enabled"}}, True),             # explicit unlimited
    (True,  512,  {"reasoning_budget": 512, "thinking": {"type": "enabled", "budget_tokens": 512}}, True),
]


@pytest.mark.parametrize("use,budget,expected,enable", CASES)
def test_body_with_switch_on(switch, use, budget, expected, enable):
    switch(True)
    assert llm._build_thinking_extra_body(use, budget) == {**expected, "chat_template_kwargs": {"enable_thinking": enable}}


@pytest.mark.parametrize("use,budget,expected,enable", CASES)
def test_body_with_switch_off_is_the_legacy_shape(switch, use, budget, expected, enable):
    switch(False)
    assert llm._build_thinking_extra_body(use, budget) == expected


# ── _stream_completion: reasoning_content used to be dropped ─────────────────

def chunk(content=None, reasoning=None, finish=None, usage=None, choices=True):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)] if choices else [], usage=usage)


class FakeStreamClient:
    def __init__(self, chunks=None, stream_error=None, fallback=None):
        self.chunks, self.stream_error, self.fallback, self.calls = chunks or [], stream_error, fallback, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            if self.stream_error:
                raise self.stream_error
            return iter(self.chunks)
        return self.fallback


def stream(client, sink=None, **kw):
    return llm._stream_completion(client, "q9", [{"role": "user", "content": "u"}], 0.6, 0.95, None, "classify",
                                  max_tokens=kw.get("max_tokens", 100), reasoning_sink=sink)


def test_reasoning_is_captured_counted_and_kept_out_of_the_answer():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=6)
    client = FakeStreamClient([
        chunk(reasoning="Let me "), chunk(reasoning="think."), chunk(content='{"a":'), chunk(content=' 1}'),
        chunk(finish="stop"), chunk(choices=False, usage=usage)])
    sink = []
    content, u, ttft, think_toks, token_count, finish = stream(client, sink)
    assert content == '{"a": 1}'                          # answer untouched
    assert sink == ["Let me ", "think."]
    assert think_toks == 2 and token_count == 4            # reasoning now counted (was always 0)
    assert finish == "stop" and u is usage


def test_without_a_sink_reasoning_is_still_counted_but_nothing_breaks():
    client = FakeStreamClient([chunk(reasoning="hmm"), chunk(content="ok", finish="stop")])
    content, _, _, think_toks, _, _ = stream(client, None)
    assert content == "ok" and think_toks == 1


def test_a_model_that_sends_no_reasoning_field_is_unaffected():
    plain = SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason="stop")   # no reasoning_content attr at all
    client = FakeStreamClient([SimpleNamespace(choices=[plain], usage=None)])
    sink = []
    content, _, _, think_toks, _, finish = stream(client, sink)
    assert content == "hi" and sink == [] and think_toks == 0 and finish == "stop"


def test_think_tags_in_content_still_counted_as_before():
    client = FakeStreamClient([chunk(content="<think>"), chunk(content="pondering"), chunk(content="</think>"),
                               chunk(content="answer", finish="stop")])
    _, _, _, think_toks, _, _ = stream(client, [])
    assert think_toks >= 2


def test_non_streaming_fallback_also_returns_reasoning():
    message = SimpleNamespace(content="answer", reasoning_content="fallback thinking")
    fallback = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage="U")
    client = FakeStreamClient(stream_error=RuntimeError("no streaming"), fallback=fallback)
    sink = []
    content, usage, *_ = stream(client, sink)
    assert content == "answer" and sink == ["fallback thinking"] and usage == "U"
    assert [c.get("stream") for c in client.calls] == [True, None]


# ── call_model ───────────────────────────────────────────────────────────────

class Verdict(BaseModel):
    decision: str
    confidence: str = "high"


GOOD = json.dumps({"decision": "CONTINUE"})


@pytest.fixture
def cm(monkeypatch, fake_model_manager):
    """call_model with config, client, logging and streaming replaced by recorders."""
    e = SimpleNamespace(stream_calls=[], stage_logs=[], thinking_logs=[], routing_budget=777, cap=1000,
                        reasoning=[], content=GOOD, finish="stop", cfg_extra={})

    def cfg(k):
        return {"base_url": "http://x/v1", "model_id": "q9", "name": "Qwen3.5-9B", "temperature": 0.6,
                "top_p": 0.9, **e.cfg_extra}

    def fake_stream(client, model_id, messages, temp, top_p, extra_body, stage, max_tokens=None, reasoning_sink=None,
                    presence_penalty=None):
        e.stream_calls.append({"extra_body": extra_body, "stage": stage, "max_tokens": max_tokens,
                               "presence_penalty": presence_penalty})
        if reasoning_sink is not None:
            reasoning_sink.extend(e.reasoning)
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20)
        return e.content, usage, 12.0, len(e.reasoning), len(e.reasoning) + 5, e.finish

    monkeypatch.setattr(llm, "get_model_config", cfg)
    monkeypatch.setattr(llm, "OpenAI", lambda **kw: object())
    monkeypatch.setattr(llm.instructor, "from_openai", lambda c, mode=None: object())
    monkeypatch.setattr(llm, "_stream_completion", fake_stream)
    monkeypatch.setattr(llm, "_get_http_timeout", lambda: 5.0)
    monkeypatch.setattr(llm, "_get_output_token_cap", lambda stage: e.cap)
    monkeypatch.setattr(llm, "_get_thinking_budget", lambda stage, *a, **k: e.routing_budget)
    monkeypatch.setattr(llm, "_thinking_control_flag", lambda name, default: True)
    monkeypatch.setattr(llm, "_log_stage_entry", lambda **k: e.stage_logs.append(k))
    monkeypatch.setattr(llm, "_write_thinking_log", lambda rd, st, tb: e.thinking_logs.append(tb))
    e.run = lambda **kw: llm.call_model("9b", [{"role": "user", "content": "u"}], Verdict, "classify", "/tmp", **kw)
    return e


def body(cm):
    return cm.stream_calls[-1]["extra_body"]


def test_routing_budget_is_now_used_when_no_explicit_budget_is_given(cm):
    """THE bug: extra_body was built from the raw budget_tokens arg (None from call_role) so this was unlimited."""
    cm.run(thinking=True)
    assert body(cm)["reasoning_budget"] == 777
    assert body(cm)["thinking"] == {"type": "enabled", "budget_tokens": 777}


def test_explicit_budget_beats_routing(cm):
    cm.run(thinking=True, budget_tokens=64)
    assert body(cm)["reasoning_budget"] == 64


def test_minus_one_is_still_the_explicit_unlimited(cm):
    cm.run(thinking=True, budget_tokens=-1)
    assert "reasoning_budget" not in body(cm) and body(cm)["thinking"] == {"type": "enabled"}


def test_routing_zero_means_thinking_off_even_if_the_node_asked_for_thinking(cm):
    cm.routing_budget = 0
    cm.run(thinking=True)
    assert body(cm)["thinking"] == {"type": "disabled"}
    assert body(cm)["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_false_is_disabled_at_both_levels(cm):
    cm.run(thinking=False)
    assert body(cm)["thinking"] == {"type": "disabled"}
    assert body(cm)["chat_template_kwargs"] == {"enable_thinking": False}


def test_model_default_applies_when_the_node_does_not_choose(cm):
    cm.cfg_extra = {"thinking": {"default_on": True}}
    cm.run()
    assert body(cm)["reasoning_budget"] == 777


def test_nowait_logit_bias_is_merged_and_skippable(cm):
    cm.cfg_extra = {"nowait_tokens": {"11": "Wait"}}
    cm.run(thinking=False)
    assert body(cm)["logit_bias"] == {"11": -100.0}
    cm.run(thinking=False, skip_nowait=True)
    assert "logit_bias" not in body(cm)


def test_result_is_parsed_and_stage_logged_ok(cm):
    assert cm.run(thinking=False).decision == "CONTINUE"
    assert cm.stage_logs[-1]["status"] == "ok" and cm.stage_logs[-1]["tokens_out"] == 20


def test_reasoning_content_is_logged_and_makes_think_ratio_real(cm):
    cm.reasoning = ["step one. ", "step two."]
    cm.run(thinking=True)
    assert cm.thinking_logs == ["step one. step two."]
    assert 0 < cm.stage_logs[-1]["think_ratio"] < 1


def test_reasoning_never_feeds_the_confidence_signal(cm):
    """Deliberate: thinking from reasoning_content is logged, but escalation still keys off <think> text in content."""
    cm.reasoning = ["<confidence>low</confidence>"]
    assert cm.run(thinking=True).confidence == "high"


def test_confidence_signal_in_content_think_block_still_works(cm):
    cm.content = "<think>ok <confidence>low</confidence></think>" + GOOD
    assert cm.run(thinking=True).confidence == "low"


def test_truncation_carries_the_thinking_so_a_runaway_is_diagnosable(cm):
    cm.reasoning = ["Wait, ", "let me reconsider... "]
    cm.content, cm.finish = "", "length"
    with pytest.raises(llm.TruncatedOutputError) as exc:
        cm.run(thinking=False)
    assert exc.value.thinking_block == "Wait, let me reconsider... "
    assert exc.value.cap == 1000 and exc.value.stage == "classify"
    assert cm.stage_logs[-1]["status"] == "truncated" and cm.stage_logs[-1]["think_ratio"] > 0


def test_truncation_merges_reasoning_with_an_open_think_tag_in_content(cm):
    cm.reasoning = ["from reasoning_content"]
    cm.content, cm.finish = "<think>from content, never closed", "length"
    with pytest.raises(llm.TruncatedOutputError) as exc:
        cm.run(thinking=True)
    assert "from reasoning_content" in exc.value.thinking_block and "from content" in exc.value.thinking_block

# ── presence_penalty: forwarded only when a model config sets one ─────────────
# (35b's presence_penalty: 1.0 "prevents infinite loops in <think>" — it was declared in models.yaml
#  for a long time but no call site read it, so it did nothing. These pin that it now reaches the wire,
#  and that a model WITHOUT one still sends nothing rather than an explicit 0.0.)

def test_call_model_forwards_the_configured_presence_penalty(cm):
    cm.cfg_extra = {"presence_penalty": 1.0}
    cm.run(thinking=False)
    assert cm.stream_calls[-1]["presence_penalty"] == 1.0


def test_call_model_passes_none_when_the_model_has_no_presence_penalty(cm):
    cm.run(thinking=False)
    assert cm.stream_calls[-1]["presence_penalty"] is None


def test_an_explicit_zero_is_forwarded_not_mistaken_for_unset(cm):
    """`if presence_penalty:` would drop a deliberate 0.0; the code must test `is not None`."""
    cm.cfg_extra = {"presence_penalty": 0.0}
    cm.run(thinking=False)
    assert cm.stream_calls[-1]["presence_penalty"] == 0.0


def _sent_kwargs(presence_penalty):
    """Run the REAL _stream_completion against a client that records what it was asked to send."""
    client = FakeStreamClient([chunk(content="ok", finish="stop")])
    llm._stream_completion(client, "q9", [{"role": "user", "content": "u"}], 0.6, 0.95, None, "classify",
                           max_tokens=100, presence_penalty=presence_penalty)
    return client.calls[0]


def test_stream_completion_sends_presence_penalty_when_set():
    assert _sent_kwargs(1.0)["presence_penalty"] == 1.0


def test_stream_completion_omits_the_field_entirely_when_unset():
    assert "presence_penalty" not in _sent_kwargs(None)


def test_stream_completion_sends_an_explicit_zero():
    assert _sent_kwargs(0.0)["presence_penalty"] == 0.0