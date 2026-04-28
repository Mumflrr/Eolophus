"""
nodes/ideation.py — 27B IQ2_S ideation (long mode only).

Broad problem space exploration. Output will be explicitly filtered
by the 9B consistency check before any planning occurs.
Thinking is off — breadth not depth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from clients.llm import call_role, compress_text
from pipeline.state import PipelineState
from schemas.ideation_output import IdeationOutput

log = logging.getLogger(__name__)

_SYSTEM = """You are an architectural ideation model. Your role is to explore
the problem space broadly and generate diverse approaches. Do NOT write code.
Do NOT produce a final plan. Generate ideas — your output will be reviewed and
filtered before any implementation decisions are made.

Be creative. Include unconventional approaches that could still potentially work. Flag tradeoffs honestly.
Precision is less important than breadth at this stage. Limit yourself to 
a maximum of 5 approaches if applicable. Use short bullet points inside the JSON strings."""


def ideation_node(state: PipelineState) -> dict:
    """
    Generate IdeationOutput from the normalised task input.
    Writes ideation.json to disk; content will be discarded after planning.
    """
    run_dir = state["run_dir"]
    task    = state.get("normalised_input") or state.get("raw_text_input", "")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"Explore approaches for this task:\n\n{task}"},
    ]

    ideation: IdeationOutput = call_role(
        role            = "ideation",
        messages        = messages,
        response_schema = IdeationOutput,
        stage           = "ideation",
        run_dir         = run_dir,
        thinking        = False,   # breadth mode — no thinking overhead
    )

    log.info(
        "Ideation: %d approaches, %d directions, %d components",
        len(ideation.approaches),
        len(ideation.architectural_directions),
        len(ideation.potential_components),
    )

    # Write to disk (will be discarded after planning to save context)
    ideation_path = str(Path(run_dir) / "ideation.json")
    Path(ideation_path).write_text(
        ideation.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "ideation_output": ideation,
        "ideation_path":   ideation_path,
    }
