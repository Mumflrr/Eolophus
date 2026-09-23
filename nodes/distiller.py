"""
nodes/distiller.py — self-improvement loop.

Generates a concise lesson if the pipeline had to iterate before succeeding.
Only fires when iteration > 0 AND final verdict is pass.

Prompt lives in config/prompts/distiller.yaml.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from clients.llm import call_role
from nodes._shared import derive_task_tags, record_escalation
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


def _load_failure_history(run_dir: str) -> list[dict]:
    """
    The non-pass verdicts from earlier correction-loop iterations, oldest first.

    distiller_node only runs its LLM call when the FINAL verdict is a pass — and a
    pass verdict says nothing about what went wrong. The failing verdicts are
    snapshotted by clients.llm.write_iteration_artifact under
    <run_dir>/iterations/<n>/verdict.json (run_dir is the turn dir for a chat turn).
    """
    base = Path(run_dir) / "iterations"
    if not base.is_dir():
        return []
    dirs = sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()),
                  key=lambda p: int(p.name))
    failed: list[dict] = []
    for d in dirs:
        try:
            v = json.loads((d / "verdict.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(v, dict) and v.get("category") not in (None, "pass"):
            failed.append({**v, "loop_iteration": int(d.name)})
    return failed


def _format_failure_history(failed: list[dict], max_chars: int = 3000) -> str:
    return json.dumps(failed, indent=2, default=str)[:max_chars]


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

    task_type: str = state.get("task_type") or "coding"

    # A lesson needs a problem to learn from — see _load_failure_history. Without
    # any failing-verdict snapshot the model only sees a passing verdict, so the
    # call can't produce anything meaningful; skip it (also covers a run whose
    # iteration counter is > 0 without there ever having been a failure).
    failed_verdicts = _load_failure_history(state["run_dir"])
    if not failed_verdicts:
        log.info(
            "Distiller skipped: iteration=%d but no failing-verdict snapshots under %s/iterations",
            iteration, state["run_dir"],
        )
        return {"pipeline_complete": state.get("pipeline_complete", True)}

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("distiller")

    lesson_output: DistilledLesson = call_role(
        role            = "distiller",
        template_vars   = {
            "iteration":   str(iteration),
            "verdict_json": verdict.model_dump_json(indent=2),
            "failed_verdicts_json": _format_failure_history(failed_verdicts),
        },
        response_schema = DistilledLesson,
        stage           = "distill",
        run_dir         = state["run_dir"],
        thinking        = False,
        profile         = profile,
        current_model_override = current_model_override,
    )

    if lesson_output.is_valuable and lesson_output.lesson_text:
        log.info("Distilled lesson: %s", lesson_output.lesson_text)
        _save_lesson(
            state, task_type, lesson_output.lesson_text,
            failure_summary=str(failed_verdicts[0].get("description") or ""),
        )
    else:
        log.debug("Distiller: fix was not universally valuable — skipping")

    result: dict[str, Any] = {"pipeline_complete": True}

    # DistilledLesson has no confidence field, so truncation is the only
    # possible trigger — deliberately not surfaced as a substantive lesson
    # (design doc §2.3: distiller learns the task-solving difference, not
    # the escalation event), just bookkeeping for the run-detail UI.
    if getattr(lesson_output, "_escalated_to", None):
        result["escalated_models"], result["escalation_history"] = \
            record_escalation(state, "distiller", lesson_output)

    return result


def _scrub_attachment_references(text: str, attachments: list) -> str:
    """
    Defense-in-depth for the attachment isolation invariant described in
    _save_lesson: replace any verbatim mention of an attached filename or
    a long verbatim substring of its content with a placeholder, so a
    lesson can never end up quoting file-specific material even if a
    critic's free-text verdict happened to reference it.

    attachments entries are the plain {filename, content} dicts built by
    _load_live_attachments / start_run's initial_state — same shape either
    way, so this works whether it's the first turn or a later replan.
    """
    if not text or not attachments:
        return text
    scrubbed = text
    for a in attachments:
        filename = a.get("filename") if isinstance(a, dict) else getattr(a, "filename", None)
        if filename and filename in scrubbed:
            scrubbed = scrubbed.replace(filename, "[attached file]")
        content = a.get("content") if isinstance(a, dict) else getattr(a, "content", None)
        if content:
            # Catch verbatim chunks of file content (>=40 chars) that got
            # quoted into the verdict text — short overlaps are too likely
            # to be coincidental (a common variable name, etc.) to scrub.
            for line in content.splitlines():
                line = line.strip()
                if len(line) >= 40 and line in scrubbed:
                    scrubbed = scrubbed.replace(line, "[attached file content]")
    return scrubbed


def _save_lesson(
    state:              PipelineState,
    task_type:          str,
    lesson_text:        str,
    issue_category_override: Optional[str] = None,
    source_chat_seq:    Optional[int]      = None,
    failure_summary:    Optional[str]      = None,
) -> None:
    """
    Construct a full Lesson object and write it to the lesson store.

    issue_category_override / source_chat_seq exist for chat_distiller_node
    (below) to reuse this same construction/write path for user-correction
    lessons, rather than duplicating the tag-derivation and write logic.
    """
    try:
        from schemas.lesson import Lesson
        from storage.lesson_store import write_lesson

        verdict   = state.get("validation_verdict")
        plan      = state.get("plan_spec")
        appraisal = state.get("appraisal_report")

        # Derive tags from plan's routing context
        tags = derive_task_tags(task_type, plan.moe_routing_context if plan else "")

        # Determine which model caught the issue
        critique = state.get("critique_record")
        model_caught = "9b"
        issue_category = issue_category_override or "other"
        if critique and critique.critic_verdicts:
            for cv in critique.critic_verdicts:
                if cv.category != "pass":
                    model_caught = cv.critic_model
                    break
        if not issue_category_override and appraisal and appraisal.issues:
            issue_category = appraisal.issues[0].category if appraisal.issues else "other"

        # ── Attachment isolation ─────────────────────────────────────────
        # Lessons must never carry file content forward — distillation is
        # meant to capture a reusable RULE ("Always await X"), not any
        # specific input data. This function's own inputs are already
        # clean: it's only ever passed verdict.description and an LLM-
        # generated lesson_text, never state["attachments"] or draft/fixed
        # code directly. The one indirect risk is verdict.description
        # itself — free text a critic model wrote, which could in principle
        # quote a fragment of an attached file by name or content if the
        # critic referenced it while explaining an issue. Scrub known
        # attachment filenames out of the summary as a defense-in-depth
        # measure; this is belt-and-suspenders, not the primary guarantee.
        # Prefer the first FAILING verdict's description (the actual problem); the
        # final verdict is a pass, whose description doesn't describe an issue.
        issue_summary = (
            (failure_summary or "")[:200]
            or (verdict.description[:200] if verdict and verdict.description else "")
            or "Pipeline required correction iterations before passing."
        )
        issue_summary = _scrub_attachment_references(issue_summary, state.get("attachments") or [])

        lesson = Lesson(
            lesson_uuid        = str(uuid.uuid4()),
            source_run_uuid    = state["run_uuid"],
            issue_summary      = issue_summary,
            resolution_pattern = lesson_text,
            example_context    = None,
            task_type          = task_type,
            tags               = list(set(tags)),
            model_caught       = model_caught,
            # Enum members must go through .value — str(Enum.member) is 'Cls.member'.
            issue_category     = str(getattr(issue_category, "value", issue_category)),
            confidence_score   = 1.0,
            times_seen         = 1,
        )

        # write_lesson returns the uuid of the row it wrote OR, when it dedups,
        # of the EXISTING row it merged into. Previously that was ignored and
        # "written" was logged with the new (never-inserted) uuid either way.
        stored_uuid = write_lesson(lesson)
        merged = stored_uuid != lesson.lesson_uuid

        if source_chat_seq is not None and not merged:
            _set_source_chat_seq(lesson.lesson_uuid, source_chat_seq)

        if merged:
            log.info("Lesson merged into existing %s (near-duplicate) — no new row", stored_uuid[:8])
        else:
            log.info("Lesson written to store: %s", lesson.lesson_uuid[:8])

    except Exception as e:
        log.warning("Failed to write lesson to store: %s", e)


def _set_source_chat_seq(lesson_uuid: str, chat_seq: int) -> None:
    """Best-effort: stamp which chat turn this lesson traces back to.
    Purely informational (see schema_additions.sql) — failure here should
    never block the lesson write itself, hence the broad except."""
    try:
        from storage.db import get_conn
        conn = get_conn()
        try:
            conn.execute(
                "UPDATE lessons SET source_chat_seq = ? WHERE lesson_uuid = ?",
                (chat_seq, lesson_uuid),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        log.warning("Failed to stamp source_chat_seq on %s: %s", lesson_uuid[:8], e)


# ── User-correction lessons (NOT YET WIRED IN) ──────────────────────────────────
#
# Distinct category from distiller_node above: that one fires on the
# EXISTING critique/validation machinery (iteration > 0, verdict == pass) —
# code-correctness rules like "always await X". This is for the case you
# described separately: the *user* corrects Claude's reasoning or choices
# mid-chat (architecture decisions, why-this-approach judgment calls), which
# has no verdict/critique_record to key off of at all.
#
# This function is a stub — NOT called from anywhere yet. The open question
# is detection: how do we decide a given user message is a correction worth
# possibly distilling, versus an ordinary follow-up? Two live options,
# discussed but not decided:
#   (a) a small/cheap LLM classifier call on every user turn — accurate,
#       but costs one extra model call per message even when nothing is
#       learned (chat-scale, so this adds up).
#   (b) a cheap heuristic (keyword/pattern match: "no,", "actually,", "why
#       did you", a question referencing a prior choice) that gates whether
#       we bother making the LLM call at all — cheaper, but will miss
#       corrections that don't match the pattern and may false-positive on
#       ordinary follow-up questions.
# Whichever is chosen, the actual distillation call and _save_lesson
# plumbing below is what it should end up calling.
#
# def chat_distiller_node(run_uuid: str, chat_history: list[dict], task_type: str) -> None:
#     """
#     Given full chat history (oldest-first, from chat_store.get_messages),
#     decide whether the latest user turn represents a correction to prior
#     reasoning/choices, and if so, distill + write a lesson via the same
#     _save_lesson() path used above, tagged issue_category="reasoning" (or
#     similar) so it's distinguishable from code-correctness lessons in the
#     Lessons UI's category filter.
#     """
#     raise NotImplementedError("detection strategy not yet decided — see note above")