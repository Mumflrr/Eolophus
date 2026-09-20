"""
nodes/ideation.py — 27B IQ2_XXS ideation (long mode only).

Broad problem space exploration. Output explicitly filtered by the 9B
consistency check before any planning occurs. Thinking is off — breadth not depth.

Prompt lives in config/prompts/ideation.yaml.

Web search moved to agentic tool calling — see clients/tools.py and
clients/llm.py's call_model_with_tools, and planner.py's module docstring
for the fuller rationale (this node's migration is identical in shape).
ideation.yaml's old {search_block} placeholder is removed as of this
migration; nothing formats into it anymore.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
from clients.tools import (
    SEARCH_HINT, SEARCH_TOOL_SCHEMA, TOOL_IMPLEMENTATIONS, format_search_notes,
)
from nodes._shared import record_escalation
from pipeline.state import PipelineState
from schemas.ideation_output import IdeationOutput

log = logging.getLogger(__name__)


def ideation_node(state: PipelineState) -> dict:
    """
    Generate IdeationOutput from the normalised task input.
    Writes ideation.json to disk; content discarded after planning.
    """
    run_dir = state["run_dir"]
    task    = state.get("normalised_input") or state.get("raw_text_input", "")

    # ── Web search (opt-in via RunRequest.use_search / ChatMessageIn's
    # replan path — see state.py's use_search field) — now offered to the
    # model as a real tool rather than pre-fetched into the prompt. See
    # planner.py's module docstring for the full rationale; identical here.
    call_tools = [SEARCH_TOOL_SCHEMA] if state.get("use_search") else None
    tool_history: list = []   # filled by call_role (tool_history_sink)

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("ideation")

    ideation: IdeationOutput = call_role(
        role            = "ideation",
        # search_hint / chat_block ALWAYS supplied ("" when unused) — see
        # planner.py's template_vars comment (unknown {placeholders} are left
        # as literal text by _safe_format).
        template_vars   = {
            "task":        task,
            "search_hint": SEARCH_HINT if call_tools else "",
            "chat_block":  "",
        },
        response_schema = IdeationOutput,
        stage           = "ideation",
        run_dir         = run_dir,
        thinking        = False,
        tools           = call_tools,
        tool_impls      = TOOL_IMPLEMENTATIONS if call_tools else None,
        tool_history_sink = tool_history,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models, escalation_history = record_escalation(state, "ideation", ideation)

    log.info(
        "Ideation: %d approaches | %d directions | %d components",
        len(ideation.approaches),
        len(ideation.architectural_directions),
        len(ideation.potential_components),
    )

    ideation_path = str(Path(run_dir) / "ideation.json")
    Path(ideation_path).write_text(
        ideation.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "ideation_output":    ideation,
        "ideation_path":      ideation_path,
        # Handed to plan_node (which appends its own results and passes the
        # combined text on to drafting). Overwritten every time ideation
        # runs so stale notes from an earlier chat turn can't leak in.
        "search_notes":       format_search_notes(tool_history),
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }