"""
nodes/distiller.py — The self-improvement loop.
Generates a concise lesson if the pipeline had to fix errors before succeeding.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from clients.llm import call_role
from pipeline.state import PipelineState

# Assuming you have this based on the docstring in critique_store.py
try:
    from storage.lesson_store import save_lesson
except ImportError:
    # Fallback stub if not implemented yet
    def save_lesson(task_type: str, tags: list[str], lesson: str) -> None:
        pass

log = logging.getLogger(__name__)

class DistilledLesson(BaseModel):
    is_valuable: bool = Field(description="True if the correction teaches a reusable architectural or syntax rule.")
    lesson_text: str = Field(description="A strict, 1-2 sentence rule on what went wrong and how to do it correctly next time.")

_DISTILLER_SYSTEM = """You are a Principal Engineer distilling knowledge.
You will be given a validation verdict containing an error and how it was resolved.
If this is a generic typo, set `is_valuable` to false.
If this is a framework-specific issue, API mismatch, or architectural flaw, set `is_valuable` to true, and write a strict 1-2 sentence rule in `lesson_text` starting with "Always" or "Never" to prevent this in the future."""

def distiller_node(state: PipelineState) -> dict:
    """
    Look at the run history. If we iterated and succeeded, extract a lesson.
    """
    iteration = state.get("iteration", 0)
    verdict = state.get("validation_verdict")
    
    # Only distill if we actually had to fix something AND we ultimately succeeded
    if iteration == 0 or not verdict or verdict.category != "pass":
        log.debug("Distiller skipped: Run either failed entirely or succeeded on the first try.")
        return {"pipeline_complete": state.get("pipeline_complete", True)}

    task_type = state.get("task_type", "coding")
    tags = state.get("classification", {}).get("tags", []) if state.get("classification") else []
    
    # We use the 9B here (role 'plan' or 'validate') because it's already hot in VRAM 
    # from the final_validate node. No model swap needed!
    messages = [
        {"role": "system", "content": _DISTILLER_SYSTEM},
        {"role": "user", "content": f"Verdict Data:\n{verdict.model_dump_json(indent=2)}"}
    ]

    log.info("Run required %d iterations. Distilling lesson...", iteration)
    
    lesson_output: DistilledLesson = call_role(
        role="validate",  # Mapped to 9B in models.yaml
        messages=messages,
        response_schema=DistilledLesson,
        stage="distill",
        run_dir=state["run_dir"],
        thinking=False,
        budget_tokens=0
    )

    if lesson_output.is_valuable and lesson_output.lesson_text:
        log.info(f"Learned new lesson: {lesson_output.lesson_text}")
        save_lesson(task_type, tags, lesson_output.lesson_text)
    else:
        log.debug("Distiller decided the fix was not universally valuable.")

    return {"pipeline_complete": True}