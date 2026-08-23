"""
nodes/ideation.py — 27B IQ2_XXS ideation (long mode only).

Broad problem space exploration. Output explicitly filtered by the 9B
consistency check before any planning occurs. Thinking is off — breadth not depth.

Prompt lives in config/prompts/ideation.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role
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

    ideation: IdeationOutput = call_role(
        role            = "ideation",
        template_vars   = {"task": task},
        response_schema = IdeationOutput,
        stage           = "ideation",
        run_dir         = run_dir,
        thinking        = False,
    )

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
        "ideation_output": ideation,
        "ideation_path":   ideation_path,
    }
