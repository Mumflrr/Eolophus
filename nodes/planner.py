"""
nodes/planner.py — 9B thinking mode planning.

Three sequential operations:
  1. Consistency check — flag and drop infeasible ideas from ideation
  2. Domain scaffold   — build MoE routing context for 35B
  3. PlanSpec          — translate viable ideas into ordered implementation plan

Injects relevant lessons from LessonL store if Phase 2 is active.
Writes planspec.json to disk; persists entire run.

Prompt lives in config/prompts/plan.yaml — no system prompts in this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role, compress_text
from pipeline.state import PipelineState
from schemas.plan_spec import PlanSpec
from schemas.lesson import LessonQuery
from storage.lesson_store import retrieve_lessons, format_lessons_for_prompt


def _get_budget(stage: str) -> int:
    from clients.llm import _get_thinking_budget
    return _get_thinking_budget(stage)


log = logging.getLogger(__name__)


def plan_node(state: PipelineState) -> dict:
    """
    Generate PlanSpec from normalized input + optional ideation.
    Retrieves relevant lessons if any are found.
    """
    run_dir       = state["run_dir"]
    task          = state.get("normalised_input") or state.get("raw_text_input", "")
    task_type     = state.get("task_type", "coding")

    # ── Compress ideation to save context window space ─────────────────────
    ideation_block = ""
    ideation = state.get("ideation_output")
    if ideation:
        raw_ideation = ideation.model_dump_json(indent=2)
        compressed   = compress_text(raw_ideation, ratio=0.5, min_tokens=200)
        ideation_block = f"\nIdeation Output (Compressed):\n{compressed}\n"

    # ── Lesson retrieval ───────────────────────────────────────────────────
    tags    = _derive_tags(task, task_type)
    lessons = retrieve_lessons(
        query=LessonQuery(
            task_type        = task_type,
            tags             = tags,
            task_description = task,
            top_k            = 3,
        ),
    )
    lessons_block = format_lessons_for_prompt(lessons)
    if lessons_block:
        lessons_block = f"\n{lessons_block}\n"

    # ── Correction context (re-plan iterations) ────────────────────────────
    correction_block = _build_correction_context(state)

    # ── Call via YAML template — no _SYSTEM constant needed ───────────────
    max_attempts = 3
    plan = None

    # Base template vars
    template_vars = {
        "task":             task,
        "ideation_block":   ideation_block,
        "lessons_block":    lessons_block,
        "correction_block": correction_block,
    }

    # Build the retry message list for subsequent attempts
    extra_messages: list[dict] = []

    for attempt in range(max_attempts):
        try:
            plan: PlanSpec = call_role(
                role            = "plan",
                template_vars   = template_vars,
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = PlanSpec,
                stage           = "plan",
                run_dir         = run_dir,
                thinking        = True,
                max_retries     = 0,
            )
            break

        except Exception as e:
            log.warning(
                "Planner JSON validation failed (attempt %d/%d): %s",
                attempt + 1, max_attempts, str(e)
            )
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Planner failed to produce valid PlanSpec after {max_attempts} attempts."
                ) from e

            # Feed exact schema error back for next attempt
            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed Pydantic validation:\n{str(e)}\n\n"
                        f"Please ensure your PlanSpec is strictly valid JSON."
                    ),
                },
            ]

    if plan.confidence == "low" and plan.clarification_question:
        log.warning("Planner halted — needs human input: %s", plan.clarification_question)
        return {
            "pipeline_halted":      True,
            "clarification_needed": plan.clarification_question,
        }

    log.info(
        "Plan: %d components | dropped=%d | routing ctx=%s",
        len(plan.implementation_order),
        len(plan.dropped_ideas),
        plan.moe_routing_context or "none",
    )

    plan_path = str(Path(run_dir) / "planspec.json")
    Path(plan_path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    return {
        "plan_spec":       plan,
        "plan_spec_path":  plan_path,
        "ideation_output": None,    # discard after planning to free context
        "relevant_lessons":lessons,
    }


def _build_correction_context(state: PipelineState) -> str:
    """Build correction feedback string for re-plan iterations."""
    iteration = state.get("iteration", 0)
    if iteration == 0:
        return ""

    verdict = state.get("validation_verdict")
    if not verdict:
        return ""

    parts = [f"\n[REPLAN — iteration {iteration}]"]
    if verdict.description:
        parts.append(f"Issue: {verdict.description}")
    if verdict.specific_issues:
        parts.append("Specific issues requiring plan changes:")
        for issue in verdict.specific_issues:
            parts.append(f"  - {issue}")
    return "\n".join(parts) + "\n"


def _derive_tags(task: str, task_type: str) -> list[str]:
    tags = [task_type]
    text = task.lower()
    tag_keywords = {
        "python":        ["python", ".py", "def ", "import "],
        "fastapi":       ["fastapi", "fast api"],
        "django":        ["django"],
        "pydantic":      ["pydantic"],
        "async":         ["async", "await", "asyncio"],
        "sqlalchemy":    ["sqlalchemy", "sql alchemy"],
        "typescript":    ["typescript", ".ts", "interface "],
        "react":         ["react", "jsx", "tsx"],
        "docker":        ["docker", "container"],
        "rest":          ["rest api", "endpoint", "route"],
        "database":      ["database", "db", "sqlite", "postgres", "mysql"],
        "testing":       ["test", "pytest", "unittest"],
        "cli":           ["cli", "command line", "argparse"],
        "class":         ["class ", "oop", "object"],
        "error_handling":["error", "exception", "try", "except"],
    }
    for tag, kws in tag_keywords.items():
        if any(kw in text for kw in kws):
            tags.append(tag)
    return list(set(tags))
