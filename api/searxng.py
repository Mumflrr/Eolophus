"""
api/searxng.py — best-effort SearXNG (Docker) lifecycle, wired into the
FastAPI app's lifespan.

Mirrors clients/search.py's own tolerance of SearXNG being unreachable
(search_web() never raises, just returns [] and logs a warning) —
nothing here ever blocks or fails startup/shutdown. This does NOT
install Docker or SearXNG — that's still setup_searxng.sh, run once by
hand. This only brings up (and optionally tears down) an
already-installed container, the same docker-compose commands
start_all.sh's start_searxng() and stop_all.sh already run — just
triggered by uvicorn instead of requiring those scripts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

_SEARXNG_COMPOSE_FILE = PROJECT_ROOT / "servers/docker-compose.searxng.yml"

# Whether to also stop the container on server shutdown. Off by default —
# `restart: unless-stopped` in the compose file is meant to let SearXNG
# outlive individual server restarts (see docker-compose.yml comments),
# so tearing it down here on every uvicorn --reload cycle would defeat
# that. Opt in explicitly if you want strict start/stop symmetry.
_SEARXNG_STOP_ON_SHUTDOWN = os.environ.get("SEARXNG_STOP_ON_SHUTDOWN", "").lower() in ("1", "true", "yes")


def _searxng_compose(*args: str) -> bool:
    """Run `docker-compose -f <file> <args>`, logging but never raising.

    Uses the hyphenated v1 CLI (docker-compose), not the v2 `docker compose`
    plugin subcommand — matches what's actually installed (see docker.io +
    docker-compose 1.29.2 on this box) and what start_all.sh/stop_all.sh
    already invoke. Switch both sides together if you later move to v2.
    """
    if not _SEARXNG_COMPOSE_FILE.exists():
        log.info(
            "SearXNG: %s not found — skipping (%s). Run setup_searxng.sh once "
            "if you want web search available.",
            _SEARXNG_COMPOSE_FILE.name, " ".join(args),
        )
        return False
    try:
        result = subprocess.run(
            ["docker-compose", "-f", str(_SEARXNG_COMPOSE_FILE), *args],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.warning(
                "SearXNG: `docker-compose %s` failed (rc=%d): %s",
                " ".join(args), result.returncode, result.stderr.strip(),
            )
            return False
        log.info("SearXNG: docker-compose %s OK", " ".join(args))
        return True
    except FileNotFoundError:
        log.info("SearXNG: docker-compose not found on PATH — skipping (%s).", " ".join(args))
        return False
    except subprocess.TimeoutExpired:
        log.warning("SearXNG: `docker-compose %s` timed out after 30s.", " ".join(args))
        return False
    except Exception as e:
        log.warning("SearXNG: unexpected error running docker-compose %s: %s", " ".join(args), e)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: bring SearXNG up in the background so a slow/failed Docker
    # call can never delay uvicorn actually starting to serve requests.
    await asyncio.to_thread(_searxng_compose, "up", "-d")
    yield
    # Shutdown: opt-in only — see _SEARXNG_STOP_ON_SHUTDOWN above.
    if _SEARXNG_STOP_ON_SHUTDOWN:
        await asyncio.to_thread(_searxng_compose, "down")
