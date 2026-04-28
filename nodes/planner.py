"""
nodes/planner.py — 9B thinking mode planning.

Three sequential operations:
  1. Consistency check — flag and drop infeasible ideas from ideation
  2. Domain scaffold   — build MoE routing context for 35B
  3. PlanSpec          — translate viable ideas into ordered implementation plan

Injects relevant lessons from LessonL store if Phase 2 is active.
Writes planspec.json to disk; persists entire run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from clients.llm import call_role, compress_text
from pipeline.state import PipelineState
from schemas.plan_spec import PlanSpec
from schemas.lesson import LessonQuery
from storage.lesson_store import retrieve_lessons, format_lessons_for_prompt


def _get_budget(stage: str) -> int:
    """Read thinking token budget for this stage from routing.yaml."""
    from clients.llm import _get_thinking_budget
    return _get_thinking_budget(stage)


log = logging.getLogger(__name__)

_SYSTEM = """You are the planning model for a software development pipeline.

You will receive a task description and optionally an IdeationOutput from
a separate model. Your job is to:

1. CONSISTENCY CHECK — Before planning, explicitly check the IdeationOutput
   (if present) for: contradictions between approaches, infeasible suggestions,
   far-fetched ideas that cannot be translated into concrete steps, and
   out-of-scope suggestions. Flag and drop these in dropped_ideas.

2. DOMAIN SCAFFOLD — Identify the technology stack, test framework, and
   execution context.

3. PLAN SPEC — Generate a precise, step-by-step implementation plan.
   Break the solution down into distinct, testable components.
   Every component must have a clear responsibility.

Reflect your confidence in the `confidence` field. If you are unsure or missing critical information, set confidence to 'low' and write a specific question to the user in `clarification_question`."""


def plan_node(state: PipelineState) -> dict:
    """
    Generate PlanSpec from normalized input + optional ideation.
    Retrieves relevant lessons if any are found.
    """
    run_dir       = state["run_dir"]
    task          = state.get("normalised_input") or state.get("raw_text_input", "")
    task_type     = state.get("task_type", "coding")
    ideation_text = ""

    # Compress ideation to save context window space
    ideation = state.get("ideation_output")
    if ideation:
        raw_ideation = ideation.model_dump_json(indent=2)
        ideation_text = compress_text(raw_ideation, ratio=0.5, min_tokens=200)

    # ── Lesson Retrieval ──
    tags = _derive_tags(task, task_type)
    lessons = retrieve_lessons(
        query = LessonQuery(task_description=task, tags=tags),
        limit = 3
    )
    lessons_context = format_lessons_for_prompt(lessons)

    messages = [{"role": "system", "content": _SYSTEM}]
    
    user_prompt = f"Task:\n{task}\n"
    if ideation_text:
        user_prompt += f"\nIdeation Output (Compressed):\n{ideation_text}\n"
    if lessons_context:
        user_prompt += f"\nLessons Learned from past runs:\n{lessons_context}\n"
        
    messages.append({"role": "user", "content": user_prompt})

    plan: PlanSpec = call_role(
        role            = "plan",
        messages        = messages,
        response_schema = PlanSpec,
        stage           = "plan",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = _get_budget("plan"),
        max_retries     = 0,
    )

    if plan.confidence == "low" and plan.clarification_question:
        log.warning("Planner halted — needs human input: %s", plan.clarification_question)
        return {
            "pipeline_halted": True,
            "clarification_needed": plan.clarification_question
        }

    log.info(
        "Plan: %d components | dropped ideas=%d | routing ctx=%s",
        len(plan.implementation_order),
        len(plan.dropped_ideas),
        plan.moe_routing_context or "none",
    )

    # Write to disk
    plan_path = str(Path(run_dir) / "planspec.json")
    Path(plan_path).write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )

    # Discard IdeationOutput from state to free context
    return {
        "plan_spec":        plan,
        "plan_spec_path":   plan_path,
        "ideation_output":  None,   # explicitly discard after planning
        "relevant_lessons": lessons,
    }


def _derive_tags(task: str, task_type: str) -> list[str]:
    """
    Derive retrieval tags from task text using simple keyword matching.
    These tags are used for lesson retrieval scoring.
    """
    tags = [task_type]
    text = task.lower()

    # Language / framework detection
    tag_keywords = {
        "python":   ["python", ".py", "def ", "import "],
        "fastapi":  ["fastapi", "fast api"],
        "django":   ["django"],
        "pydantic": ["pydantic"],
        "async":    ["async", "await", "asyncio"],
        "sqlalchemy":["sqlalchemy", "sql alchemy"],
        "typescript":["typescript", ".ts", "interface "],
        "react":    ["react", "jsx", "tsx"],
        "docker":   ["docker", "container"],
        "rest":     ["rest api", "endpoint", "route"],
        "database": ["database", "db", "sqlite", "postgres", "mysql"],
        "testing":  ["test", "pytest", "unittest"],
        "cli":      ["cli", "command line", "argparse"],
        "class":    ["class ", "oop", "object"],
        "error_handling": ["error", "exception", "try", "except"],
    }

    for tag, kws in tag_keywords.items():
        if any(kw in text for kw in kws):
            tags.append(tag)

    return list(set(tags))