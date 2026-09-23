"""tests/live/thinking_control.py — OFFLINE tests of the server probe (fake and local-loopback servers, never your real one)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.live import thinking_control as C


def reply(tokens, reasoning="", content="yes", finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}],
            "usage": {"completion_tokens": tokens}}


THINKS = reply(500, reasoning="Let me think about whether 17 is prime... " * 10)
QUIET = reply(2)


def fake_server_that_honours(switch):
    """A `post` that stops thinking only when `switch(body)` is true — i.e. a server honouring exactly one mechanism."""
    seen = []

    def post(url, body):
        seen.append(body)
        return QUIET if switch(body) else THINKS
    post.seen = seen
    return post


def run_all(post):
    return [(label, C.probe(post, "http://x/v1", "q9", extra, 100)) for label, extra in C.VARIANTS]


# ── probe ────────────────────────────────────────────────────────────────────

def test_probe_sends_the_extra_fields_and_the_prompt():
    post = fake_server_that_honours(lambda b: False)
    C.probe(post, "http://x/v1/", "q9", {"chat_template_kwargs": {"enable_thinking": False}}, 123)
    body = post.seen[0]
    assert body["model"] == "q9" and body["max_tokens"] == 123
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["content"] == C.PROMPT


def test_probe_posts_to_chat_completions_without_a_double_slash():
    urls = []
    C.probe(lambda url, body: urls.append(url) or QUIET, "http://x/v1/", "q9", {}, 10)
    assert urls == ["http://x/v1/chat/completions"]


def test_probe_reads_reasoning_from_the_separate_field():
    r = C.probe(lambda u, b: THINKS, "http://x/v1", "q9", {}, 10)
    assert r["tokens"] == 500 and r["reasoning_chars"] > 100 and not C.is_off(r)


def test_probe_reads_inline_think_tags_too():
    inline = reply(300, reasoning="", content="<think>lots of pondering here</think>yes")
    r = C.probe(lambda u, b: inline, "http://x/v1", "q9", {}, 10)
    assert r["reasoning_chars"] == len("lots of pondering here") and r["content"] == "yes"


def test_probe_flags_a_capped_response():
    assert C.probe(lambda u, b: reply(1500, finish="length"), "http://x/v1", "q9", {}, 10)["truncated"] is True


def test_a_short_quiet_answer_counts_as_off_and_a_long_or_reasoned_one_does_not():
    assert C.is_off({"tokens": 2, "reasoning_chars": 0})
    assert not C.is_off({"tokens": 2, "reasoning_chars": 50})
    assert not C.is_off({"tokens": 400, "reasoning_chars": 0})


# ── analyze: one scenario per kind of server ─────────────────────────────────

def test_server_honouring_only_chat_template_kwargs():
    text = " ".join(C.analyze(run_all(fake_server_that_honours(lambda b: "chat_template_kwargs" in b))))
    assert "`thinking: disabled` is IGNORED" in text
    assert "enable_thinking=false WORKS" in text and "Keep routing.yaml" in text


def test_server_honouring_the_legacy_thinking_key():
    text = " ".join(C.analyze(run_all(fake_server_that_honours(lambda b: b.get("thinking", {}).get("type") == "disabled"))))
    assert "`thinking: disabled` DOES work" in text


def test_server_honouring_only_reasoning_budget():
    text = " ".join(C.analyze(run_all(fake_server_that_honours(lambda b: b.get("reasoning_budget") == 0))))
    assert "Only reasoning_budget=0 works" in text


def test_server_honouring_nothing_gets_the_server_side_advice():
    text = " ".join(C.analyze(run_all(fake_server_that_honours(lambda b: False))))
    assert "NONE of the per-request switches" in text and "--reasoning-budget 0" in text


def test_a_server_that_never_thinks_is_reported_as_not_the_cause():
    text = " ".join(C.analyze(run_all(fake_server_that_honours(lambda b: True))))
    assert "NOT what is making your stages long" in text


def test_a_runaway_is_reproduced_and_called_out():
    post = lambda url, body: reply(1500, reasoning="x" * 5000, finish="length")
    assert any("runaway, reproduced" in line for line in C.analyze(run_all(post)))


# ── end to end over real HTTP ────────────────────────────────────────────────

@pytest.fixture
def server():
    """A real local HTTP server that honours only chat_template_kwargs, exercising http_post + main()."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            payload = QUIET if "chat_template_kwargs" in body else THINKS
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", requests
    srv.shutdown()


def test_main_end_to_end(server, capsys):
    url, requests = server
    assert C.main(["--base-url", url, "--model-id", "q9"]) == 0
    out = capsys.readouterr().out
    assert len(requests) == len(C.VARIANTS)
    assert out.count("[THINKING ON]") == 3 and out.count("[thinking OFF]") == 2
    assert "enable_thinking=false WORKS" in out


def test_main_reports_an_unreachable_server(capsys):
    assert C.main(["--base-url", "http://127.0.0.1:1/v1", "--model-id", "q9"]) == 2
    assert "Could not reach the server" in capsys.readouterr().err