"""
api/routing_config.py — load/save config/routing.yaml (thinking_budgets).
"""

from __future__ import annotations

import yaml

from api.paths import PROJECT_ROOT


def load_routing_config() -> dict:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def save_routing_config(cfg: dict) -> None:
    p = PROJECT_ROOT / "config" / "routing.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
