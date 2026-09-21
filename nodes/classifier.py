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
from enum import Enum
from pathlib import Path

from clients.llm import (
    call_role, load_prompt, _safe_format, TruncatedOutputError, EscalationNeeded,
)
from pipeline.state import PipelineState
from pipeline.routers import select_profile
from storage.critique_store import write_run
from schemas.task_classification import TaskClassification
from typing import Optional
import openai

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


def _as_db_str(value, default: str = "auto") -> str:
    """
    Coerce a mode / task_type / complexity value to the plain string
    storage.write_run expects.

    These fields can be a str (pinned by the caller), an Enum (from
    TaskClassification), or None (nothing pinned and the model didn't set
    one). Enums must go through .value: str(Mode.CODING) yields
    "Mode.CODING", which would silently store the wrong text. None falls
    back to "auto", the same placeholder runs.start_run already writes
    for these columns before classification has run.
    """
    if value is None:
        return default
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


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
    # human_in_the_loop gates ESCALATION itself (design doc §2.6, RE-
    # corrected): True (default) means call_role must not silently call a
    # bigger model — it raises EscalationNeeded instead, which this node
    # now turns into a request for human clarification rather than a
    # yes/no escalation prompt (see the except block below). Passed
    # straight through as require_confirmation.
    human_in_the_loop = state.get("human_in_the_loop", True)

    max_attempts  = 3
    classification: Optional[TaskClassification] = None
    extra_messages: list[dict] = []
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])

    for attempt in range(max_attempts):
        try:
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
                profile         = profile,
                current_model_override = current_model_override,
                require_confirmation = human_in_the_loop,
            )
            break

        except EscalationNeeded as esc:
            # Ask for help instead of escalating (design doc §2.6, RE-
            # corrected): a stage that wants to escalate no longer asks
            # "should I try a bigger model?" — it asks the person a real,
            # open-ended clarifying question and re-runs on the SAME
            # model with their answer folded in, exactly like any other
            # low-confidence clarification. There is no separate
            # escalation-confirmation halt, no escalation_confirmation.json,
            # and no interrupt() call here at all: this except block just
            # produces a `classification` with confidence="low" and a real
            # clarification_question, then falls through to the existing
            # "remaining low confidence" handling below (same code path a
            # model-authored low-confidence result already goes through),
            # which is what actually routes to clarify_node. No model
            # bump ever happens on resume — see esc.trigger branches below.
            if esc.trigger == "low_confidence":
                # esc.result is already a validly-parsed TaskClassification
                # (the low-confidence result call_role intercepted before
                # it could auto-escalate) — reuse it as-is rather than
                # fabricating one. If the model already wrote its own
                # clarification_question, that's real signal about what's
                # actually ambiguous — keep it. Only fall back to a
                # generic prompt when the model didn't give us one to work
                # with (schema allows null even at confidence=low).
                if esc.result is None:
                    # call_role always attaches result for trigger=="low_confidence";
                    # fail loudly if that invariant ever breaks.
                    raise RuntimeError(
                        "EscalationNeeded(low_confidence) carried no result"
                    ) from esc
                esc_result: TaskClassification = esc.result
                if not esc_result.clarification_question:
                    esc_result = esc_result.model_copy(update={
                        "clarification_question": (
                            "I'm not fully confident in this classification "
                            "and could use more direction before continuing — "
                            "what would help clarify the task?"
                        ),
                    })
                classification = esc_result
                log.info(
                    "Classify: low confidence on '%s' — asking for clarification "
                    "instead of escalating to '%s'",
                    esc.current_model_id, esc.next_model_id,
                )
                break

            # esc.trigger == "truncation": no parsed result to fall back
            # on (the call never finished), so there's no classification
            # object to attach a clarification_question to. Re-run once,
            # pinned to the SAME model with escalation disabled — if it
            # truncates again, that's a normal TruncatedOutputError for
            # the loop's own handler a few lines down to catch and
            # propagate to graph.py's _wrap_node_for_truncation_retry
            # (the existing "retry with a higher token cap" flow), which
            # is a genuinely different UI/flow from clarification and the
            # right place for "the model literally ran out of room" to
            # land — not a question the person could usefully answer in
            # words. See the TruncatedOutputError handler a few lines
            # below for the identical re-run-then-propagate pattern this
            # mirrors.
            log.info(
                "Classify: truncated on '%s' — retrying same model instead "
                "of escalating to '%s'",
                esc.current_model_id, esc.next_model_id,
            )
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

        except openai.APIConnectionError as e:
            log.warning(
                "Classifier: model server unreachable (attempt %d/%d, likely "
                "still loading/swapping) — retrying without feedback: %s",
                attempt + 1, max_attempts, str(e)
            )
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Classifier: model server never became reachable after {max_attempts} attempts."
                ) from e
            # No extra_messages update — nothing to correct, just retry.

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

    # The loop either breaks with a classification or raises on its last
    # attempt; make that explicit (and narrow Optional[TaskClassification]).
    if classification is None:
        raise RuntimeError("Classifier exited retry loop without a result")

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
    # escalated_models/escalation_history are populated ONLY by the
    # human_in_the_loop=False (set-and-forget) path now: call_role auto-
    # escalated internally exactly as before and attached
    # _escalated_from/_escalated_to onto the result — read those back
    # here. This folds into run-level state so (a) a later re-classify
    # resumes from the escalated model rather than restarting at the
    # bottom, and (b) the run-detail UI can show the purple escalation
    # badge (design doc §2.7). This is NOT a substantive lesson —
    # escalation_history is infra bookkeeping, kept separate from
    # whatever distiller.py writes to the lesson store, per §2.3's
    # "distiller learns the substantive difference, not the escalation
    # event."
    #
    # When human_in_the_loop=True, the EscalationNeeded except block above
    # never escalates at all anymore — it asks for clarification instead
    # (see that block's comment) — so escalated_models/escalation_history
    # simply stay as whatever the caller passed in; this block is a no-op
    # for that path.
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
    # By this point, any need to escalate has already been resolved one
    # way or another: either the EscalationNeeded except block above
    # turned it into a clarification question (human_in_the_loop=True —
    # classification now reflects that, with confidence=="low" and a real
    # question set), human_in_the_loop=False already auto-escalated
    # internally above with no one to ask, or there was no ladder to
    # escalate to in the first place. Any confidence=="low" surviving all
    # of that still needs a final decision:
    #   human_in_the_loop=True  — halt and surface the question, whether
    #     it came from the model's own classification or from the
    #     ask-for-help path above.
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
            log.warning("Classifier confidence=low — halting for clarification: %s",
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
        mode            = _as_db_str(final_mode),
        task_type       = _as_db_str(final_task_type),
        complexity      = _as_db_str(classification.complexity),
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