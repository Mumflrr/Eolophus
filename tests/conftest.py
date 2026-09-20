"""
Shared pytest setup for the Eolophus test suite.

  pip install -r requirements-test.txt
  pytest                                  # everything (no model / network needed)
  pytest --cov --cov-report=term-missing  # with coverage (config in .coveragerc)

These tests never talk to a real model or to SearXNG: the OpenAI client, the
search backend and model_manager are all replaced with scripted fakes, so the
suite is fast and deterministic. They DO import your real modules, so a
regression in clients/llm.py, the nodes, routers.py or the prompt YAMLs shows
up here. Anything that needs a RUNNING model lives in tests/live/ instead, and is
never collected by pytest (see TESTING.md).
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]      # project root: tests/ lives directly under it
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    return ROOT


@pytest.fixture
def run_dir(tmp_path) -> str:
    """A throwaway runs/<uuid>/ directory for nodes that write artefacts."""
    return str(tmp_path)


@pytest.fixture
def fake_model_manager(monkeypatch):
    """
    call_model / call_model_with_tools do `from clients.model_manager import
    ensure_model_loaded` at call time. Swap the module so no test can ever try
    to start (or wait for) a real llama-server.
    """
    mod = types.ModuleType("clients.model_manager")
    mod.ensure_model_loaded = lambda key: False
    mod.current_model = lambda: None
    mod.stop_all = lambda: None
    monkeypatch.setitem(sys.modules, "clients.model_manager", mod)
    return mod