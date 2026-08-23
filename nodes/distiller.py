"""
nodes/distiller.py — self-improvement loop.

Generates a concise lesson if the pipeline had to iterate before succeeding.
Only fires when iteration > 0 AND final verdict is pass.

Prompt lives in config/prompts/distiller.yaml.
"""

from __future__ import annotations

import logging
import uuid
from pydantic import BaseModel, Field

from clients.llm import call_role
from pipeline.state import PipelineState

log = logging.getLogger(__name__)


class DistilledLesson(BaseModel):
    is_valuable: bool = Field(
        description=(
            "True if the correction teaches a reusable architectural or "
            "framework-specific rule. False for generic typos or formatting."
        )
    )
    lesson_text: str = Field(
        description=(
            "A strict 1-2 sentence rule starting with 'Always' or 'Never'. "
            "Names the specific framework/API/pattern. "
            "Empty string if is_valuable=False."
        )
    )


def distiller_node(state: PipelineState) -> dict:
    """
    Look at the run history. If we iterated and succeeded, extract a lesson.
    """
    iteration = state.get("iteration", 0)
    verdict   = state.get("validation_verdict")

    # Only distill if we iterated AND ultimately succeeded
    if iteration == 0 or not verdict or verdict.category != "pass":
        log.debug(
            "Distiller skipped: iteration=%d, verdict=%s",
            iteration,
            getattr(verdict, "category", "none"),
        )
        return {"pipeline_complete": state.get("pipeline_complete", True)}

    task_type = state.get("task_type", "coding")

    lesson_output: DistilledLesson = call_role(
        role            = "distiller",
        template_vars   = {
            "iteration":   str(iteration),
            "verdict_json": verdict.model_dump_json(indent=2),
        },
        response_schema = DistilledLesson,
        stage           = "distill",
        run_dir         = state["run_dir"],
        thinking        = False,
    )

    if lesson_output.is_valuable and lesson_output.lesson_text:
        log.info("Distilled lesson: %s", lesson_output.lesson_text)
        _save_lesson(state, task_type, lesson_output.lesson_text)
    else:
        log.debug("Distiller: fix was not universally valuable — skipping")

    return {"pipeline_complete": True}


def _save_lesson(state: PipelineState, task_type: str, lesson_text: str) -> None:
    """
    Construct a full Lesson object and write it to the lesson store.
    """
    try:
        from schemas.lesson import Lesson
        from storage.lesson_store import write_lesson

        verdict   = state.get("validation_verdict")
        plan      = state.get("plan_spec")
        appraisal = state.get("appraisal_report")

        # Derive tags from plan's routing context
        tags = [task_type]
        if plan and plan.moe_routing_context:
            ctx = plan.moe_routing_context.lower()
            for kw in ["python", "fastapi", "async", "django", "typescript",
                       "react", "database", "rest", "docker", "testing", "pydantic"]:
                if kw in ctx:
                    tags.append(kw)

        # Determine which model caught the issue
        critique = state.get("critique_record")
        model_caught = "9b"
        issue_category = "other"
        if critique and critique.critic_verdicts:
            for cv in critique.critic_verdicts:
                if cv.category != "pass":
                    model_caught = cv.critic_model
                    break
        if appraisal and appraisal.issues:
            issue_category = appraisal.issues[0].category if appraisal.issues else "other"

        lesson = Lesson(
            lesson_uuid        = str(uuid.uuid4()),
            source_run_uuid    = state["run_uuid"],
            issue_summary      = (
                verdict.description[:200] if verdict and verdict.description
                else "Pipeline required correction iterations before passing."
            ),
            resolution_pattern = lesson_text,
            example_context    = None,
            task_type          = task_type,
            tags               = list(set(tags)),
            model_caught       = model_caught,
            issue_category     = str(issue_category),
            confidence_score   = 1.0,
            times_seen         = 1,
        )

        write_lesson(lesson)
        log.info("Lesson written to store: %s", lesson.lesson_uuid[:8])

    except Exception as e:
        log.warning("Failed to write lesson to store: %s", e)
