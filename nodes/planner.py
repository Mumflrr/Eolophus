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

log = logging.getLogger(__name__)

_SYSTEM = """You are the planning model for a software development pipeline.

You will receive a task description and optionally an IdeationOutput from
a separate model. Your job is to:

1. CONSISTENCY CHECK — Before planning, explicitly check the IdeationOutput
   (if present) for: contradictions between approaches, infeasible suggestions,
   far-fetched ideas that cannot be translated into concrete steps, and
   out-of-scope suggestions. Flag and drop these in dropped_ideas.

2. DOMAIN SCAFFOLD — Identify the technology domains, patterns, languages,
   and frameworks relevant to this task. This becomes the moe_routing_context
   that primes expert routing in the 35B MoE executor.

3. PLAN — Translate viable ideas into a concrete, ordered PlanSpec.
   Be precise: function signatures, data structures, implementation order,
   edge cases, and assumptions. Only viable ideas survive to this stage.

When you are confident in your plan, emit: <confidence>high</confidence>
in your reasoning before your final answer.
"""


def plan_node(state: PipelineState) -> dict:
    """
    Produce a PlanSpec from the task input and optional IdeationOutput.
    Uses thinking mode for consistency checking and planning precision.
    """
    run_dir       = state["run_dir"]
    task          = state.get("normalised_input") or state.get("raw_text_input", "")
    ideation      = state.get("ideation_output")
    iteration     = state.get("iteration", 0)
    task_type     = state.get("task_type", "coding")
    classification = state.get("classification")

    # Retrieve relevant lessons (Phase 2 only — returns [] in Phase 1)
    tags = _derive_tags(task, task_type)
    lessons = retrieve_lessons(LessonQuery(
        task_type = task_type,
        tags      = tags,
    ))
    lessons_block = format_lessons_for_prompt(lessons)

    # Build user message
    user_parts = []

    if lessons_block:
        user_parts.append(lessons_block)
        user_parts.append("")

    user_parts.append(f"Task:\n{task}")

    if ideation:
        # Compress IdeationOutput prose before injecting (not code)
        ideation_text = ideation.model_dump_json(indent=2)
        ideation_compressed = compress_text(
            ideation_text, ratio=0.5, min_tokens=200
        )
        user_parts.append(f"\nIdeationOutput (from 27B — apply consistency check):\n{ideation_compressed}")

    # Include correction feedback if this is a re-plan
    if iteration > 0:
        verdict = state.get("validation_verdict")
        if verdict:
            feedback_parts = []
            if verdict.description:
                feedback_parts.append(f"Previous plan issue: {verdict.description}")
            if verdict.specific_issues:
                issues_str = "\n".join(f"  - {i}" for i in verdict.specific_issues)
                feedback_parts.append(f"Specific issues to address:\n{issues_str}")
            if feedback_parts:
                feedback_text = "\n".join(feedback_parts)
                # Compress correction history on later iterations
                if iteration > 1:
                    feedback_text = compress_text(feedback_text, ratio=0.6, min_tokens=100)
                user_parts.append(f"\n[CORRECTION FEEDBACK — iteration {iteration}]\n{feedback_text}")

    user_message = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    plan: PlanSpec = call_role(
        role            = "plan",
        messages        = messages,
        response_schema = PlanSpec,
        stage           = "plan",
        run_dir         = run_dir,
        thinking        = True,
        budget_tokens   = 4096,
    )

    log.info(
        "Plan: %d components, %d edge cases, %d dropped, confidence=%s",
        len(plan.components),
        len(plan.edge_cases),
        len(plan.dropped_ideas),
        plan.confidence_in_plan,
    )

    # Write PlanSpec to disk — persists entire run
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

    for tag, keywords in tag_keywords.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    return list(set(tags))
