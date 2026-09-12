"""
clients/search.py — SearXNG search client.

A single search_web() function nodes can call, plus a formatter for
folding results into a prompt. Used by nodes/planner.py's plan_node and
nodes/ideation.py's ideation_node, gated behind state.get("use_search"):

    from clients.search import search_web, format_results_for_prompt

    if state.get("use_search"):
        results = search_web(query)
        if results:
            task = f"{task}\n\n{format_results_for_prompt(results)}"

use_search IS declared on PipelineState (state.py) and IS read by both
of those nodes today.

Historical note, kept because the same class of bug can recur: this
function raises nothing and logs at warning level on every failure it
detects — connection refused, timeout, non-JSON response, a 403 from
SearXNG's format=json being disabled — SO A SILENT "search never
happened, no warning, no error" symptom does NOT mean this file is
broken. That combination previously meant use_search was never making
it into the state dict a given code path built at all: state.get() on a
TypedDict with total=False returns None for an absent key with no error,
call_role/call_model never got invoked, so there was nothing to log.
Concretely, this bit chat follow-ups: post_chat_message's ChatMessageIn
schema had no use_search field, so _run_chat_replan's turn_state never
set the key regardless of what the ORIGINAL run had asked for or
what a person might want to toggle on a later turn. Fixed in server.py
(ChatMessageIn.use_search) and api.js (sendChatMessage's useSearch arg) —
if a similar silent gap shows up again, check every place that BUILDS a
state dict passed into app_graph.invoke() (start_run's initial_state,
_run_chat_replan's turn_state, and any future ones) actually sets
use_search, not just this file or the two nodes that read it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)


class SearchResult(BaseModel):
    title:   str
    url:     str
    content: str = ""   # SearXNG's snippet/summary field


class SearchError(Exception):
    """Raised when SearXNG can't be reached or returns something unusable."""


def _searxng_url() -> str:
    # Mirrors server.py's GET /config/searxng default exactly, so the UI's
    # displayed URL and the URL actually queried never drift apart.
    return os.environ.get("SEARXNG_URL", "http://localhost:8888")


def search_web(
    query: str,
    max_results: int = 5,
    timeout: float = 10.0,
) -> list[SearchResult]:
    """
    Query SearXNG's JSON API and return up to max_results results.

    Returns an empty list (never raises) on any failure — a broken or
    misconfigured search backend should degrade the pipeline's answer
    quality, not crash a run. Failures are logged at warning level so
    they're visible without being fatal.

    Common failure mode: SearXNG returns 403 on format=json unless
    `search.formats` in settings.yml has been edited to include `json`
    (it's disabled by default in stock SearXNG for anti-scraping
    reasons). That shows up here as an httpx.HTTPStatusError.
    """
    query = query.strip()
    if not query:
        return []

    base_url = _searxng_url().rstrip("/")
    params = {
        "q":      query,
        "format": "json",
    }

    try:
        resp = httpx.get(f"{base_url}/search", params=params, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            log.warning(
                "SearXNG at %s returned 403 for format=json — JSON output is "
                "likely disabled in settings.yml. Add `json` to `search.formats` "
                "and restart SearXNG. Query was: %r",
                base_url, query,
            )
        else:
            log.warning("SearXNG returned HTTP %s for query %r: %s",
                        e.response.status_code, query, e)
        return []
    except httpx.ConnectError as e:
        log.warning(
            "Couldn't reach SearXNG at %s (query=%r): %s. Is it running, and "
            "is SEARXNG_URL set correctly? Check GET /config/searxng.",
            base_url, query, e,
        )
        return []
    except httpx.TimeoutException as e:
        log.warning("SearXNG request timed out after %ss (query=%r): %s",
                    timeout, query, e)
        return []
    except Exception as e:
        log.warning("Unexpected error querying SearXNG (query=%r): %s", query, e)
        return []

    try:
        data = resp.json()
    except ValueError as e:
        log.warning("SearXNG response for %r wasn't valid JSON: %s", query, e)
        return []

    raw_results = data.get("results", [])
    if not raw_results:
        log.info("SearXNG returned zero results for query %r", query)
        return []

    results: list[SearchResult] = []
    for r in raw_results[:max_results]:
        url = r.get("url")
        title = r.get("title")
        if not url or not title:
            continue
        results.append(SearchResult(
            title=title,
            url=url,
            content=(r.get("content") or "")[:500],
        ))

    return results


def format_results_for_prompt(results: list[SearchResult]) -> str:
    """
    Render search results as a plain-text block suitable for folding into
    a node's task/user prompt. Returns "" for an empty list so callers can
    unconditionally append it without an extra guard.
    """
    if not results:
        return ""

    lines = ["[Web search results]"]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.url}")
        if r.content:
            lines.append(f"   {r.content}")
    return "\n".join(lines)