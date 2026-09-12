"""
nodes/classifier.py — 9B task classification with confidence + clarification.

If confidence=low and clarification_question is set, the pipeline halts
immediately and returns the question to the caller.

Base system prompt lives in config/prompts/classify.yaml.
Pinned-mode variants are handled inline (Mode B) since they require
conditional system prompt selection — the YAML system is used for the
unpinned case only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import (
    call_role, load_prompt, _safe_format, TruncatedOutputError, EscalationNeeded,
)
from langgraph.types import interrupt
from pipeline.state import PipelineState
from pipeline.routers import select_profile
from storage.critique_store import write_run
from schemas.task_classification import TaskClassification
from typing import Optional

log = logging.getLogger(__name__)


# ── Pinned-mode system prompts ────────────────────────────────────────────────
# Used when the caller has forced mode and/or task_type.
# The base system (from classify.yaml) is used for the unpinned case.

_SYSTEM_PINNED_MODE = """You are a task classifier for a local LLM pipeline.
The MODE has been pinned by the user — do not change it.
Determine: task_type, complexity, decompose, confidence, clarification_question.

Set confidence=low only when the task is genuinely ambiguous in a way that
would cause the wrong output. Most tasks should be high or medium confidence.
"""

_SYSTEM_PINNED_BOTH = """You are a task classifier for a local LLM pipeline.
The MODE and TASK TYPE have been pinned by the user — do not change them.
Determine: complexity, decompose, confidence, clarification_question.
"""


def classify_node(state: PipelineState) -> dict:
    run_dir          = state["run_dir"]
    task             = state.get("normalised_input") or state.get("raw_text_input", "")
    pinned_mode      = state.get("mode")
    pinned_task_type = state.get("task_type")

    # ── Choose system prompt and build messages ────────────────────────────
    if pinned_mode and pinned_task_type:
        # Mode B: both pinned — use inline system
        pin_note = f"[PINNED] mode={pinned_mode}, task_type={pinned_task_type}\n\n"
        messages = [
            {"role": "system", "content": _SYSTEM_PINNED_BOTH},
            {"role": "user",   "content": f"{pin_note}Task to classify:\n\n{task}"},
        ]
    elif pinned_mode:
        # Mode B: mode pinned — use inline system
        pin_note = f"[PINNED] mode={pinned_mode}\n\n"
        messages = [
            {"role": "system", "content": _SYSTEM_PINNED_MODE},
            {"role": "user",   "content": f"{pin_note}Task to classify:\n\n{task}"},
        ]
    else:
        # Mode A: YAML-driven (unpinned — most common path)
        messages = None

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("classify")
    # human_in_the_loop now gates ESCALATION itself (design doc §2.6,
    # corrected): True (default) means call_role must ask before calling
    # a bigger model, not just before the final halt once the ladder's
    # exhausted. Passed straight through as require_confirmation — see
    # EscalationNeeded's docstring in clients/llm.py for the full
    # ask-then-decide flow implemented in the except block below.
    human_in_the_loop = state.get("human_in_the_loop", True)

    max_attempts  = 3
    classification = None
    extra_messages: list[dict] = []
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])

    for attempt in range(max_attempts):
        try:
            classification: TaskClassification = call_role(
                role            = "classify",
                messages        = messages,
                template_vars   = {"task": task} if messages is None else None,
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = TaskClassification,
                stage           = "classify",
                run_dir         = run_dir,
                thinking        = False,
                max_retries     = 0,
                profile         = profile,
                current_model_override = current_model_override,
                require_confirmation = human_in_the_loop,
            )
            break

        except EscalationNeeded as esc:
            # Ask before escalating (design doc §2.6, corrected). interrupt()
            # pauses this node's execution — same mechanism clarify_node
            # uses — and resumes right here with whatever answer the
            # person gave via POST /clarify (or an equivalent confirm
            # endpoint). See clarify_node's docstring in pipeline/graph.py
            # for how interrupt()/Command(resume=...) actually works.
            answer = interrupt({
                "question": (
                    f"Stage 'classify' wants to escalate from '{esc.current_model_id}' "
                    f"to '{esc.next_model_id}' ({esc.trigger.replace('_', ' ')}). Proceed?"
                ),
                "kind":          "escalation_confirmation",
                "stage":         esc.stage,
                "from_model":    esc.current_model_id,
                "to_model":      esc.next_model_id,
                "trigger":       esc.trigger,
            })
            confirmed = str(answer).strip().lower() in ("y", "yes", "true", "1")

            if confirmed:
                log.info("Escalation confirmed for classify: %s → %s",
                          esc.current_model_id, esc.next_model_id)
                classification = call_role(
                    role            = "classify",
                    messages        = messages,
                    template_vars   = {"task": task} if messages is None else None,
                    extra_messages  = extra_messages if extra_messages else None,
                    response_schema = TaskClassification,
                    stage           = "classify",
                    run_dir         = run_dir,
                    thinking        = False,
                    max_retries     = 0,
                    current_model_override = esc.next_model_id,
                    allow_escalation = False,   # one confirmed step only — no further auto-walk
                )
                escalated_models["classify"] = esc.next_model_id
                escalation_history.append({
                    "stage":      "classify",
                    "from_model": esc.current_model_id,
                    "to_model":   esc.next_model_id,
                    "trigger":    esc.trigger,
                    "iteration":  state.get("iteration", 0),
                    "confirmed":  True,
                })
                break
            else:
                log.info("Escalation declined for classify — proceeding on '%s' as-is",
                          esc.current_model_id)
                if esc.result is not None:
                    # Low-confidence trigger: esc.result is the already-
                    # parsed low-confidence TaskClassification — use it
                    # as final rather than re-calling anything.
                    classification = esc.result
                    break
                # Truncation trigger has no parsed result to fall back to.
                # Re-run once more, pinned to the SAME model with
                # allow_escalation=False, so a still-truncating result
                # propagates as a normal TruncatedOutputError for
                # graph.py's _wrap_node_for_truncation_retry to catch
                # (manual higher-cap retry), instead of looping back into
                # another confirmation prompt for the same declined step.
                classification = call_role(
                    role            = "classify",
                    messages        = messages,
                    template_vars   = {"task": task} if messages is None else None,
                    extra_messages  = extra_messages if extra_messages else None,
                    response_schema = TaskClassification,
                    stage           = "classify",
                    run_dir         = run_dir,
                    thinking        = False,
                    max_retries     = 0,
                    current_model_override = esc.current_model_id,
                    allow_escalation = False,
                )
                break

        except TruncatedOutputError as exc:
            # Item 6 fix (design doc §4): this call site previously let a
            # TruncatedOutputError propagate with no record of whatever
            # partial content the model produced before hitting its cap —
            # call_role/call_model already extract thinking_block/
            # partial_answer onto the exception (see TruncatedOutputError's
            # docstring in clients/llm.py), but nothing here was writing
            # them anywhere, and classify_node has no post-loop write of
            # its own to reuse. This IS a general pattern (plan_node has
            # the identical bare `raise` a few lines down its own loop,
            # and any future node with a similar retry loop would hit the
            # same gap) — so the actual fix lives in the graph wrapper,
            # not here: pipeline/graph.py's _wrap_node_for_truncation_retry
            # now writes a generic <node_name>_partial.json next to
            # truncated.json before calling interrupt(), using exactly the
            # same exc.thinking_block/exc.partial_answer this bare `raise`
            # already propagates. Nothing below needs its own try/except-
            # and-write — re-raising as before is sufficient; the wrapper
            # at registration time (get_graph() in pipeline/graph.py)
            # catches this for every node uniformly, classify included.
            #
            # Also note: call_role's internal escalation walk (clients/
            # llm.py) already tried the next model on classify's ladder
            # once before this exception could even reach here — if we're
            # seeing TruncatedOutputError at all, the ladder is either
            # already exhausted or escalation was disabled, so retrying
            # with the same messages here would not help. Let it propagate.
            raise

        except Exception as e:
            log.warning(
                "Classifier JSON validation failed (attempt %d/%d): %s",
                attempt + 1, max_attempts, str(e)
            )
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Classifier failed after {max_attempts} attempts."
                ) from e

            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed Pydantic validation:\n{str(e)}\n\n"
                        f"Please try again with strict JSON compliance."
                    ),
                },
            ]
            # For Mode B (pinned), rebuild messages without extra_messages
            # (extra_messages is appended by call_role)

    final_mode      = pinned_mode      or classification.mode
    final_task_type = pinned_task_type or classification.task_type

    # ── Profile resolution ──────────────────────────────────────────────────
    # requested_profile is set by server.py from the caller's request (or
    # "auto" if unspecified/explicitly "auto"). "auto" resolves HERE, once,
    # from classification's own complexity/decompose fields — see
    # pipeline/routers.py's select_profile() and design doc §4 item 2.
    # mode (final_mode above) plays no part in this resolution; it remains
    # an informational field on TaskClassification only.
    requested_profile = state.get("requested_profile") or "auto"
    if requested_profile == "auto":
        resolved_profile = select_profile(classification)
        log.info("Profile: auto → %s (complexity=%s decompose=%s)",
                  resolved_profile, classification.complexity, classification.decompose)
    else:
        resolved_profile = requested_profile

    log.info(
        "Classification: mode=%s type=%s complexity=%s decompose=%s confidence=%s profile=%s",
        final_mode, final_task_type,
        classification.complexity, classification.decompose,
        classification.confidence, resolved_profile,
    )

    # ── Escalation bookkeeping ────────────────────────────────────────────
    # Two paths populate escalated_models/escalation_history:
    #   1. human_in_the_loop=True: the EscalationNeeded except block above
    #      already appended an entry (with confirmed=True) when the person
    #      said yes. Nothing further to do here in that case.
    #   2. human_in_the_loop=False (require_confirmation=False passed to
    #      call_role): call_role auto-escalated internally exactly as
    #      before and attached _escalated_from/_escalated_to onto the
    #      result — read those back here, same as the original
    #      implementation. This folds into run-level state so (a) a later
    #      re-classify resumes from the escalated model rather than
    #      restarting at the bottom, and (b) the run-detail UI can show
    #      the purple escalation badge (design doc §2.7). This is NOT a
    #      substantive lesson — escalation_history is infra bookkeeping,
    #      kept separate from whatever distiller.py writes to the lesson
    #      store, per §2.3's "distiller learns the substantive difference,
    #      not the escalation event."
    if "classify" not in escalated_models:
        escalated_to_attr = getattr(classification, "_escalated_to", None)
        if escalated_to_attr:
            escalated_from_attr = getattr(classification, "_escalated_from", None)
            escalated_models["classify"] = escalated_to_attr
            escalation_history.append({
                "stage":      "classify",
                "from_model": escalated_from_attr,
                "to_model":   escalated_to_attr,
                "trigger":    "low_confidence" if classification.confidence != "low" else "truncation",
                "iteration":  state.get("iteration", 0),
                "confirmed":  False,   # set-and-forget path — never asked
            })
            log.info("Classify escalated %s → %s (set-and-forget)", escalated_from_attr, escalated_to_attr)

    # ── Remaining low confidence ────────────────────────────────────────────
    # By this point, escalation has already been resolved one way or
    # another: either the person confirmed and we escalated (classification
    # now reflects the escalated model's output), the person declined (we
    # accepted the current result as final), there was no ladder to
    # escalate to at all, or human_in_the_loop=False already auto-escalated
    # internally with no one to ask. Any confidence=="low" surviving all of
    # that still needs a final decision:
    #   human_in_the_loop=True  — halt and surface the question. Either the
    #     person just declined the escalation offer (so halting instead is
    #     the natural next step), or there was nothing left on the ladder
    #     to even offer.
    #   human_in_the_loop=False (set-and-forget) — no one to ask, ever, by
    #     definition. Escalation already happened automatically above if a
    #     ladder existed; if confidence is still low after that, there's no
    #     halting option left, so proceed with the best-effort result
    #     instead. clarification_question is dropped from the returned
    #     state (not just left unread) so route_after_classify's
    #     `confidence=="low" and question present` check — the actual halt
    #     condition — can't fire for a set-and-forget run.
    clarification_question = classification.clarification_question

    if classification.confidence == "low" and clarification_question:
        if human_in_the_loop:
            log.warning("Classifier confidence=low (escalation declined or unavailable): %s",
                        clarification_question)
        else:
            log.warning(
                "Classifier confidence=low after automatic escalation, but "
                "human_in_the_loop=False (set-and-forget) — proceeding "
                "best-effort instead of halting. Original question was: %s",
                clarification_question,
            )
            clarification_question = None

    resolved = classification.model_copy(update={
        "mode":      final_mode,
        "task_type": final_task_type,
    })

    write_run(
        run_uuid        = state["run_uuid"],
        mode            = final_mode,
        task_type       = final_task_type,
        complexity      = classification.complexity,
        is_sub_spec     = state.get("is_sub_spec", False),
        parent_run_uuid = state.get("parent_run_uuid"),
    )

    classification_path = str(Path(run_dir) / "classification.json")
    Path(classification_path).write_text(
        classification.model_dump_json(indent=2), encoding="utf-8"
    )

    return {
        "classification":         resolved,
        "mode":                   final_mode,
        "task_type":              final_task_type,
        "decompose":              classification.decompose,
        "classifier_confidence":  classification.confidence,
        "clarification_question": clarification_question,
        "profile":                resolved_profile,
        "requested_profile":      requested_profile,
        "escalated_models":       escalated_models,
        "escalation_history":     escalation_history,
    }