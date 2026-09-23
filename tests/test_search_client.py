"""clients/search.py — SearXNG client. search_web() must NEVER raise; every failure is a logged []."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from clients import search


def _response(status=200, json_data=None, text=None):
    req = httpx.Request("GET", "http://localhost:8888/search")
    if json_data is not None:
        return httpx.Response(status, json=json_data, request=req)
    return httpx.Response(status, text=text if text is not None else "", request=req)


@pytest.fixture
def http(monkeypatch):
    """Replace httpx.get; set http.response or http.error, inspect http.calls."""
    box = SimpleNamespace(calls=[], response=None, error=None)

    def fake_get(url, params=None, timeout=None):
        box.calls.append({"url": url, "params": params, "timeout": timeout})
        if box.error:
            raise box.error
        return box.response

    monkeypatch.setattr(search.httpx, "get", fake_get)
    return box


# ── happy path ───────────────────────────────────────────────────────────────

def test_returns_parsed_results(http):
    http.response = _response(json_data={"results": [
        {"title": "T1", "url": "http://a", "content": "snippet one"},
        {"title": "T2", "url": "http://b", "content": "snippet two"},
    ]})
    out = search.search_web("fastapi latest")
    assert [(r.title, r.url, r.content) for r in out] == [
        ("T1", "http://a", "snippet one"), ("T2", "http://b", "snippet two")]
    assert http.calls[0]["params"] == {"q": "fastapi latest", "format": "json"}
    assert http.calls[0]["url"].endswith("/search")


def test_max_results_limits_output(http):
    http.response = _response(json_data={"results": [
        {"title": f"T{i}", "url": f"http://{i}"} for i in range(10)]})
    assert len(search.search_web("q", max_results=3)) == 3
    assert len(search.search_web("q")) == 5           # default


def test_content_is_truncated_to_500_chars(http):
    http.response = _response(json_data={"results": [{"title": "T", "url": "http://a", "content": "x" * 900}]})
    assert len(search.search_web("q")[0].content) == 500


def test_missing_content_defaults_to_empty(http):
    http.response = _response(json_data={"results": [{"title": "T", "url": "http://a", "content": None}]})
    assert search.search_web("q")[0].content == ""


def test_entries_without_url_or_title_are_skipped(http):
    http.response = _response(json_data={"results": [
        {"title": "no url"}, {"url": "http://no-title"}, {"title": "ok", "url": "http://ok"}]})
    assert [r.title for r in search.search_web("q")] == ["ok"]


def test_query_is_stripped(http):
    http.response = _response(json_data={"results": []})
    search.search_web("  hello  ")
    assert http.calls[0]["params"]["q"] == "hello"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_query_returns_empty_without_any_http_call(http, blank):
    assert search.search_web(blank) == []
    assert http.calls == []


# ── every failure mode degrades to [] and logs ───────────────────────────────

def test_403_logs_a_hint_about_json_format(http, caplog):
    http.response = _response(status=403)
    caplog.set_level(logging.WARNING, logger="clients.search")
    assert search.search_web("q") == []
    assert "403" in caplog.text and "json" in caplog.text.lower()


def test_other_http_status_returns_empty(http, caplog):
    http.response = _response(status=500)
    caplog.set_level(logging.WARNING, logger="clients.search")
    assert search.search_web("q") == []
    assert "500" in caplog.text


def test_connection_error_returns_empty(http, caplog):
    http.error = httpx.ConnectError("refused")
    caplog.set_level(logging.WARNING, logger="clients.search")
    assert search.search_web("q") == []
    assert "Couldn't reach SearXNG" in caplog.text


def test_timeout_returns_empty(http, caplog):
    http.error = httpx.ReadTimeout("slow")
    caplog.set_level(logging.WARNING, logger="clients.search")
    assert search.search_web("q", timeout=3.0) == []
    assert "timed out after 3.0s" in caplog.text


def test_unexpected_exception_returns_empty(http):
    http.error = RuntimeError("something odd")
    assert search.search_web("q") == []


def test_non_json_body_returns_empty(http, caplog):
    http.response = _response(text="<html>captcha</html>")
    caplog.set_level(logging.WARNING, logger="clients.search")
    assert search.search_web("q") == []
    assert "valid JSON" in caplog.text


def test_zero_results_returns_empty(http):
    http.response = _response(json_data={"results": []})
    assert search.search_web("q") == []


def test_missing_results_key_returns_empty(http):
    http.response = _response(json_data={})
    assert search.search_web("q") == []


# ── configuration ────────────────────────────────────────────────────────────

def test_default_url(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    assert search._searxng_url() == "http://localhost:8888"


def test_url_from_env_and_trailing_slash_is_stripped(http, monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searx.example:9000/")
    http.response = _response(json_data={"results": []})
    search.search_web("q")
    assert http.calls[0]["url"] == "http://searx.example:9000/search"


# ── prompt formatting ────────────────────────────────────────────────────────

def test_format_empty_list_is_empty_string():
    assert search.format_results_for_prompt([]) == ""


def test_format_renders_numbered_block_and_omits_blank_snippets():
    out = search.format_results_for_prompt([
        search.SearchResult(title="One", url="http://1", content="first"),
        search.SearchResult(title="Two", url="http://2"),
    ])
    assert out.splitlines() == [
        "[Web search results]", "1. One", "   http://1", "   first", "2. Two", "   http://2"]