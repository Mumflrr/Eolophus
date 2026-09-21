"""
config/loader.py — single cached loader for models.yaml and routing.yaml.

Previously each of clients/llm.py, clients/model_manager.py, and
pipeline/routers.py implemented its own "open a YAML file and cache it"
function (three separate module-level cache dicts, one per file). llm.py
additionally reopened and reparsed routing.yaml, uncached, on every single
call for thinking budgets / HTTP timeout / output caps.

This module is the one place either file is read. Everything else imports
from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_CONFIG_DIR = Path(__file__).parent

_models_cache: dict = {}
_routing_cache: dict = {}


def get_models_config() -> dict:
    """Parsed contents of config/models.yaml, cached after first read."""
    if not _models_cache:
        with open(_CONFIG_DIR / "models.yaml") as f:
            _models_cache.update(yaml.safe_load(f))
    return _models_cache


def get_routing_config() -> dict:
    """
    Parsed contents of config/routing.yaml, cached after first read.

    Unlike get_models_config(), a missing/unparsable routing.yaml degrades
    to {} rather than raising — routing.yaml only holds tuning values
    (thinking budgets, timeouts, output caps) that the three getters below
    already have sane defaults for, so a bad file shouldn't be able to
    break every model call. This mirrors the original per-function
    try/except behaviour, just in one place. A failed read is NOT cached,
    so the next call retries the file (e.g. if it was mid-write).
    """
    if not _routing_cache:
        try:
            with open(_CONFIG_DIR / "routing.yaml") as f:
                _routing_cache.update(yaml.safe_load(f) or {})
        except (OSError, yaml.YAMLError):
            return {}
    return _routing_cache


def get_thinking_budget(stage: str, complexity: str = "moderate") -> int:
    """
    thinking_budgets.<stage>.<complexity> from routing.yaml, default 2048.

    stage_cfg is checked with `is None`, not truthiness — routing.yaml
    uses 0 to mean "explicitly no thinking" for several stages (bugfix,
    validate, classify, draft_pass2, ...), and `int(x) if x else 2048`
    treats a configured 0 as "not configured," silently reading it back
    as 2048.
    """
    stage_cfg = get_routing_config().get("thinking_budgets", {}).get(stage)
    if stage_cfg is None:
        return 2048
    if isinstance(stage_cfg, dict):
        return stage_cfg.get(complexity, stage_cfg.get("moderate", 2048))
    return int(stage_cfg)


def get_thinking_control_flag(name: str, default: bool) -> bool:
    """routing.yaml thinking_control.<name> (bool), or `default` if absent."""
    val = (get_routing_config().get("thinking_control") or {}).get(name)
    return default if val is None else bool(val)


def get_http_timeout() -> float:
    """http.timeout_seconds from routing.yaml, default 7200s."""
    return float(get_routing_config().get("http", {}).get("timeout_seconds", 7200))


def get_output_token_cap(stage: str) -> Optional[int]:
    """output_token_caps.<stage> from routing.yaml, or None (unbounded)."""
    cap = get_routing_config().get("output_token_caps", {}).get(stage)
    return int(cap) if cap else None

def reload_config() -> None:
    """Drop both cached configs; the next get_*_config() call rereads from disk."""
    _models_cache.clear()
    _routing_cache.clear()