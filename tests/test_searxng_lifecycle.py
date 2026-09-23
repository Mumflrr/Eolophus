"""api/searxng.py — best-effort SearXNG container lifecycle. Nothing here may ever block or fail startup."""
from __future__ import annotations

import asyncio
import logging
import subprocess
from types import SimpleNamespace

import pytest

from api import searxng as S


@pytest.fixture
def compose_file(tmp_path, monkeypatch):
    f = tmp_path / "docker-compose.searxng.yml"
    f.write_text("services: {}\n")
    monkeypatch.setattr(S, "_SEARXNG_COMPOSE_FILE", f)
    return f


@pytest.fixture
def docker(monkeypatch):
    box = SimpleNamespace(calls=[], result=SimpleNamespace(returncode=0, stderr=""), error=None)

    def fake_run(cmd, **kw):
        box.calls.append((cmd, kw))
        if box.error:
            raise box.error
        return box.result

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    return box


def test_success_runs_docker_compose_with_the_file(compose_file, docker):
    assert S._searxng_compose("up", "-d") is True
    cmd, kw = docker.calls[0]
    assert cmd == ["docker-compose", "-f", str(compose_file), "up", "-d"]
    assert kw["timeout"] == 30 and kw["capture_output"] is True


def test_missing_compose_file_skips_without_running_anything(tmp_path, monkeypatch, docker, caplog):
    monkeypatch.setattr(S, "_SEARXNG_COMPOSE_FILE", tmp_path / "nope.yml")
    caplog.set_level(logging.INFO, logger="api.searxng")
    assert S._searxng_compose("up", "-d") is False
    assert docker.calls == [] and "not found" in caplog.text


def test_nonzero_exit_returns_false_and_logs_stderr(compose_file, docker, caplog):
    docker.result = SimpleNamespace(returncode=1, stderr="port is already allocated")
    caplog.set_level(logging.WARNING, logger="api.searxng")
    assert S._searxng_compose("up", "-d") is False
    assert "port is already allocated" in caplog.text


def test_docker_compose_not_installed(compose_file, docker):
    docker.error = FileNotFoundError("docker-compose")
    assert S._searxng_compose("up", "-d") is False


def test_timeout(compose_file, docker, caplog):
    docker.error = subprocess.TimeoutExpired(cmd="docker-compose", timeout=30)
    caplog.set_level(logging.WARNING, logger="api.searxng")
    assert S._searxng_compose("up", "-d") is False
    assert "timed out" in caplog.text


def test_unexpected_error_is_swallowed(compose_file, docker):
    docker.error = OSError("boom")
    assert S._searxng_compose("up", "-d") is False


# ── lifespan ────────────────────────────────────────────────────────────────

def _run_lifespan(monkeypatch, stop_on_shutdown, helper_result=True):
    calls = []
    monkeypatch.setattr(S, "_searxng_compose", lambda *a: calls.append(a) or helper_result)
    monkeypatch.setattr(S, "_SEARXNG_STOP_ON_SHUTDOWN", stop_on_shutdown)

    async def main():
        async with S.lifespan(None):
            calls.append(("<serving>",))

    asyncio.run(main())
    return calls


def test_lifespan_starts_searxng_then_serves_and_leaves_it_running(monkeypatch):
    assert _run_lifespan(monkeypatch, stop_on_shutdown=False) == [("up", "-d"), ("<serving>",)]


def test_lifespan_stops_searxng_on_shutdown_only_when_opted_in(monkeypatch):
    assert _run_lifespan(monkeypatch, stop_on_shutdown=True) == [("up", "-d"), ("<serving>",), ("down",)]


def test_lifespan_still_serves_when_searxng_fails_to_start(monkeypatch):
    assert ("<serving>",) in _run_lifespan(monkeypatch, stop_on_shutdown=False, helper_result=False)