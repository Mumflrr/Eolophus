"""
nodes/planner.py — 9B thinking mode planning.

Three sequential operations:
  1. Consistency check — flag and drop infeasible ideas from ideation
  2. Domain scaffold   — build MoE routing context for 35B
  3. PlanSpec          — translate viable ideas into ordered implementation plan

Injects relevant lessons from LessonL store if Phase 2 is active.
Writes planspec.json to disk; persists entire run.

Prompt lives in config/prompts/plan.yaml — no system prompts in this file.

Web search moved to agentic tool calling (see clients/tools.py,
clients/llm.py's call_model_with_tools) — plan.yaml's {search_block}
placeholder, _extract_search_query(), and config/prompts/search_query.yaml
are all removed as of this migration. The model now writes its own
keyword query as a tool-call argument when it decides search would help,
instead of Python pre-fetching results before the call based on a
use_search flag and gluing them into the prompt as text. use_search still
gates whether the tool is OFFERED to the model at all (via tools=), but
no longer guarantees a search happened — that's now the model's call.

Two follow-ups to that migration (see clients/tools.py's docstring):
  - SEARCH_HINT is added to the prompt whenever the tool is offered, so the
    model is actually told it can search (plan.yaml's user_template has a
    {search_hint} slot). Without it a 9B given a "planning model" prompt
    rarely calls a tool nobody mentioned.
  - The tool-call history is captured via call_role's tool_history_sink and
    returned as state["search_notes"], so the drafting stage sees what was
    found. call_role alone returns only the PlanSpec and drops the results.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role, compress_text, TruncatedOutputError, EscalationNeeded
from clients.tools import (
    SEARCH_HINT, SEARCH_TOOL_SCHEMA, TOOL_IMPLEMENTATIONS, format_search_notes,
)
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

    # ── Web search — now agentic tool calling, not prompt-stuffing ─────────
    # use_search gates whether the search_web tool is OFFERED to the model
    # at all (see RunRequest.use_search / ChatMessageIn.use_search in
    # server.py, and state.py's use_search field). It no longer guarantees
    # a search happens — the model sees the tool and decides for itself
    # whether/what to search, the same way a person would. If the model
    # doesn't think search is needed for this task, it just won't call it,
    # which is the correct behaviour (forcing a search on every plan call
    # regardless of whether the task needs current info was never
    # actually desirable, it was just the only option prompt-stuffing
    # allowed for).
    call_tools = [SEARCH_TOOL_SCHEMA] if state.get("use_search") else None
    # Filled in place by call_role (tool_history_sink) with whatever the
    # model searched/got back, so it can be handed on to drafting below.
    tool_history: list = []

    # ── Call via YAML template — no _SYSTEM constant needed ───────────────
    max_attempts = 3
    plan = None

    # Base template vars — search_block removed; plan.yaml's user_template
    # must have its {search_block} placeholder removed too (see that file).
    template_vars = {
        "task":             task,
        "ideation_block":   ideation_block,
        "lessons_block":    lessons_block,
        "correction_block": correction_block,
        # ALWAYS supplied (empty when unused): _safe_format leaves unknown
        # {placeholders} as literal text, so omitting either of these puts
        # the raw string "{search_hint}" / "{chat_block}" in the prompt.
        # chat_block is a vestige — chat history is folded into
        # normalised_input by api/routers/chat.py — kept as "" so
        # plan.yaml can keep its slot.
        "search_hint":      SEARCH_HINT if call_tools else "",
        "chat_block":       "",
    }

    # Build the retry message list for subsequent attempts
    extra_messages: list[dict] = []

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("plan")
    # human_in_the_loop gates escalation itself (design doc §2.6, RE-
    # corrected) — see classifier.py's identical comment: True means
    # call_role raises EscalationNeeded instead of silently escalating,
    # and this node turns that into a request for clarification rather
    # than a yes/no escalation prompt (see the except block below).
    human_in_the_loop = state.get("human_in_the_loop", True)
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])

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
                tools           = call_tools,
                tool_impls      = TOOL_IMPLEMENTATIONS if call_tools else None,
                tool_history_sink = tool_history,
                profile         = profile,
                current_model_override = current_model_override,
                require_confirmation = human_in_the_loop,
            )
            break

        except EscalationNeeded as esc:
            # Ask for help instead of escalating (design doc §2.6, RE-
            # corrected) — see classifier.py's identical block for the
            # full rationale. No more yes/no escalation prompt, no
            # interrupt() call here, no escalation_confirmation.json:
            # this except block turns the escalation attempt into a real,
            # open-ended clarification_question on `plan` and falls
            # through to the existing "remaining low confidence" handling
            # below, which is what actually halts via clarify_node. No
            # model bump ever happens on resume.
            if esc.trigger == "low_confidence":
                # esc.result is already a validly-parsed PlanSpec (the
                # low-confidence result call_role intercepted before it
                # could auto-escalate) — reuse it as-is. Keep the model's
                # own clarification_question if it wrote one; only
                # substitute a generic prompt when it didn't.
                plan = esc.result
                if not plan.clarification_question:
                    plan = plan.model_copy(update={
                        "clarification_question": (
                            "I'm not fully confident in this plan and could "
                            "use more direction before continuing — what "
                            "would help clarify the task?"
                        ),
                    })
                log.info(
                    "Plan: low confidence on '%s' — asking for clarification "
                    "instead of escalating to '%s'",
                    esc.current_model_id, esc.next_model_id,
                )
                break

            # esc.trigger == "truncation": no parsed result to attach a
            # clarification_question to. Re-run once, pinned to the SAME
            # model with escalation disabled — a still-truncating result
            # propagates as a normal TruncatedOutputError for the loop's
            # own handler below to catch and hand to
            # pipeline/graph.py's _wrap_node_for_truncation_retry (the
            # existing "retry with a higher token cap" flow), which is
            # the right place for "the model ran out of room" to land —
            # not a question the person could usefully answer in words.
            log.info(
                "Plan: truncated on '%s' — retrying same model instead "
                "of escalating to '%s'",
                esc.current_model_id, esc.next_model_id,
            )
            plan = call_role(
                role            = "plan",
                template_vars   = template_vars,
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = PlanSpec,
                stage           = "plan",
                run_dir         = run_dir,
                thinking        = True,
                max_retries     = 0,
                tools           = call_tools,
                tool_impls      = TOOL_IMPLEMENTATIONS if call_tools else None,
                tool_history_sink = tool_history,
                current_model_override = esc.current_model_id,
                allow_escalation = False,
            )
            break



        except TruncatedOutputError:
            # See classifier.py's identical guard/comment — not a
            # validation failure, so the "please output valid JSON" retry
            # below won't help. Propagating lets pipeline/graph.py's
            # _wrap_node_for_truncation_retry catch it; that wrapper now
            # also writes a generic <node_name>_partial.json (item 6 fix)
            # from exc.thinking_block/exc.partial_answer before
            # interrupting, so plan_node doesn't need its own partial-save
            # logic here either. call_role's internal escalation walk
            # (clients/llm.py) already tried the next model on plan's
            # ladder once before this could reach here.
            raise

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

    # ── Escalation bookkeeping ──────────────────────────────────────────────
    # escalated_models/escalation_history are populated ONLY by the
    # human_in_the_loop=False (set-and-forget) path now — see
    # classifier.py's identical block for the full rationale. When
    # human_in_the_loop=True, the EscalationNeeded except block above
    # never escalates anymore, so this is a no-op for that path.
    if "plan" not in escalated_models:
        escalated_to_attr = getattr(plan, "_escalated_to", None)
        if escalated_to_attr:
            escalated_from_attr = getattr(plan, "_escalated_from", None)
            escalated_models["plan"] = escalated_to_attr
            escalation_history.append({
                "stage":      "plan",
                "from_model": escalated_from_attr,
                "to_model":   escalated_to_attr,
                "trigger":    "low_confidence" if plan.confidence != "low" else "truncation",
                "iteration":  state.get("iteration", 0),
                "confirmed":  False,
            })
            log.info("Plan escalated %s → %s (set-and-forget)", escalated_from_attr, escalated_to_attr)

    # ── Remaining low confidence — see classifier.py's identical comment ──
    # for the full rationale. By this point, any need to escalate has
    # already been resolved: either the EscalationNeeded except block
    # above turned it into a clarification question, human_in_the_loop=
    # False already auto-escalated with no one to ask, or there was no
    # ladder at all.
    if plan.confidence == "low" and plan.clarification_question:
        if human_in_the_loop:
            log.warning("Planner halted — needs human input: %s", plan.clarification_question)
            return {
                "pipeline_halted":      True,
                "clarification_needed": plan.clarification_question,
                "escalated_models":     escalated_models,
                "escalation_history":   escalation_history,
            }
        log.warning(
            "Planner confidence=low after automatic escalation, but "
            "human_in_the_loop=False (set-and-forget) — proceeding "
            "best-effort instead of halting. Original question was: %s",
            plan.clarification_question,
        )

    log.info(
        "Plan: %d components | dropped=%d | routing ctx=%s",
        len(plan.implementation_order),
        len(plan.dropped_ideas),
        plan.moe_routing_context or "none",
    )

    plan_path = str(Path(run_dir) / "planspec.json")
    Path(plan_path).write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    # Notes handed to drafting. Overwritten on every plan pass (never left
    # untouched) so a chat follow-up on a reused checkpoint can't inherit the
    # previous turn's results. Earlier notes are carried forward only when
    # this pass is a continuation of the same question: ideation just ran
    # (ideation_output is still set — cleared below), or this is a
    # validation-loop replan (iteration > 0; chat turns restart at 0).
    carried = ""
    if ideation or state.get("iteration", 0) > 0:
        carried = state.get("search_notes") or ""
    search_notes = format_search_notes(tool_history, carried=carried)

    return {
        "plan_spec":       plan,
        "plan_spec_path":  plan_path,
        "ideation_output": None,    # discard after planning to free context
        "search_notes":    search_notes,
        "relevant_lessons":lessons,
        "escalated_models":     escalated_models,
        "escalation_history":   escalation_history,
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