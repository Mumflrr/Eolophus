"""
clients/tools.py — tool schemas + implementations for agentic tool calling.

Replaces the old prompt-stuffing pattern (search_block in plan.yaml /
ideation.yaml, _extract_search_query in planner.py) with real OpenAI-style
tool calling: the model sees a tool schema, decides whether and what to
call, and clients.llm.call_model_with_tools() runs the loop.

Confirmed working against Qwen3.5-9B via llama.cpp (--jinja,
--host 127.0.0.1) on 2026-09-10 — see test_tool_calling.py. Native
tool_calls come back structured (not prose), so no ReAct-style text
parsing is needed for this model family.

Adding a new tool:
  1. Add a schema dict below, following SEARCH_TOOL_SCHEMA's shape.
  2. Add a matching entry to TOOL_IMPLEMENTATIONS: name -> callable that
     takes the parsed arguments dict and returns a string (the content
     that goes back to the model as the tool result).
  3. Pass tools=[..., YOUR_SCHEMA] into call_role()/call_model_with_tools()
     at whichever node call site should have access to it. Nothing here
     grants a node search "automatically" — see call_role's tools= param.
"""

from __future__ import annotations

import logging

from clients.search import search_web, format_results_for_prompt

log = logging.getLogger(__name__)


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current, up-to-date information via SearXNG. "
            "Use this when you need information that may have changed "
            "recently or that you're not confident about from training "
            "alone — current events, recent facts, specific up-to-date "
            "details. You can call this more than once if the first "
            "results aren't enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A short keyword search-engine query, 3-8 words — "
                        "what a person would type into a search box, not a "
                        "full sentence or instruction."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _run_search_web(args: dict) -> str:
    """
    Tool implementation for search_web. Takes the model's parsed tool-call
    arguments, runs the actual search, and returns the text that goes back
    to the model as the tool result message.

    search_web() itself never raises (see clients/search.py's docstring —
    it logs at warning level and returns [] on any failure), so this stays
    equally best-effort: a SearXNG outage degrades the model's answer, it
    doesn't crash the tool-call loop. format_results_for_prompt([]) returns
    "" for an empty list, so the model gets an explicit (if unhelpful)
    empty result rather than the loop erroring out.
    """
    query = (args or {}).get("query", "")
    results = search_web(query)
    if not results:
        log.info("search_web tool call returned zero results (query=%r)", query)
        return "No results found."
    return format_results_for_prompt(results)


# name -> callable(args: dict) -> str
TOOL_IMPLEMENTATIONS = {
    "search_web": _run_search_web,
}