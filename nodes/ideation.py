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
from clients.tools import SEARCH_TOOL_SCHEMA, TOOL_IMPLEMENTATIONS
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


    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("ideation")

    ideation: IdeationOutput = call_role(
        role            = "ideation",
        template_vars   = {"task": task},
        response_schema = IdeationOutput,
        stage           = "ideation",
        run_dir         = run_dir,
        thinking        = False,
        tools           = call_tools,
        tool_impls      = TOOL_IMPLEMENTATIONS if call_tools else None,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(ideation, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(ideation, "_escalated_from", None)
        escalated_models["ideation"] = escalated_to_attr
        escalation_history.append({
            "stage":      "ideation",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",   # IdeationOutput has no confidence field
            "iteration":  state.get("iteration", 0),
        })
        log.info("Ideation escalated %s → %s", escalated_from_attr, escalated_to_attr)

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
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }