"""
nodes/bugfixer.py — DeepCoder 14B bug fix and idiomatic review.

Receives: DraftOutput + AppraisalReport (if available) + PlanSpec.
Performs:
  1. Independent bug finding (corpus-trained pattern matching)
  2. Targeted fixes from AppraisalReport
  3. Idiomatic corrections and code quality polish
Produces: FixedOutput — what the critique ensemble receives.

Non-thinking mode — precise repair, not reasoning. Budget from routing.yaml.
Prompt lives in config/prompts/bugfix.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

from clients.llm import call_role, write_iteration_artifact, TruncatedOutputError
from pipeline.guards import check_fixed_output_present
from pipeline.state import PipelineState
from schemas.execution import FixedOutput

log = logging.getLogger(__name__)


def apply_diff(original_code: str, edits: list) -> str:
    """Apply SearchReplaceBlock edits to draft code in-place."""
    modified = original_code
    for edit in edits:
        if edit.search_text in modified:
            modified = modified.replace(edit.search_text, edit.replace_text, 1)
        else:
            log.warning("Diff Engine: could not find exact search_text in code — skipping edit")
    return modified


def bugfix_node(state: PipelineState) -> dict:
    """
    Apply bug fixes and idiomatic corrections to the draft.
    Writes fixed.json to disk.
    """
    run_dir   = state["run_dir"]
    draft     = state.get("draft_output")
    appraisal = state.get("appraisal_report")
    plan      = state.get("plan_spec")
    iteration = state.get("iteration", 0)

    if not draft:
        raise ValueError("bugfix_node: missing draft_output")

    # ── Build template vars ────────────────────────────────────────────────
    appraisal_block = ""
    if appraisal:
        appraisal_block = (
            f"AppraisalReport (apply all identified fixes):\n"
            f"{appraisal.model_dump_json(indent=2)}"
        )
    else:
        appraisal_block = "[No AppraisalReport available — apply your own review only]"

    correction_block = ""
    if iteration > 0:
        verdict = state.get("validation_verdict")
        if verdict and verdict.specific_issues:
            issues_str = "\n".join(f"  - {i}" for i in verdict.specific_issues)
            correction_block = (
                f"\n[Validator feedback from iteration {iteration}]:\n{issues_str}"
            )

    template_vars = {
        "plan_json":       plan.model_dump_json(indent=2) if plan else "not available",
        "draft_json":      draft.model_dump_json(indent=2),
        "appraisal_block": appraisal_block,
        "correction_block":correction_block,
    }

    # Was a single call relying only on Instructor's internal max_retries=1
    # to self-correct malformed JSON. search_text/replace_text carry verbatim
    # source code — quotes, backslashes, embedded newlines — which is exactly
    # the content most prone to JSON-escaping mistakes (e.g. a model emitting
    # an unescaped quote or dropping a comma between adjacent string fields,
    # which breaks the JSON parser itself rather than failing Pydantic
    # validation). When Instructor's one internal retry also fails, it raises
    # InstructorRetryException, which — unlike classify_node/plan_node/
    # draft_node — nothing here caught, so it propagated straight out of the
    # node, past _wrap_node_for_truncation_retry (which only catches
    # TruncatedOutputError), and crashed the whole run with status="error"
    # and no failure_reason.
    #
    # Add the same outer retry loop those other nodes already use: catch the
    # broader exception class (not just rely on Instructor's own internal
    # retry), feed the exact parse/validation error back to the model, and
    # try again up to max_attempts times before finally giving up.
    max_attempts = 3
    fixed = None
    extra_messages: list[dict] = []

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("bugfix")

    for attempt in range(max_attempts):
        try:
            fixed: FixedOutput = call_role(
                role            = "bugfix",
                template_vars   = template_vars,
                extra_messages  = extra_messages if extra_messages else None,
                response_schema = FixedOutput,
                stage           = "bugfix",
                run_dir         = run_dir,
                thinking        = False,
                max_retries     = 1,
                profile         = profile,
                current_model_override = current_model_override,
            )
            break

        except TruncatedOutputError:
            # See classifier.py's identical guard — a token-cap truncation
            # isn't a JSON validation failure, so retrying with the same cap
            # plus a "please output valid JSON" nudge won't help, and
            # wrapping it in RuntimeError below would hide it from
            # pipeline/graph.py's generic truncation-retry wrapper.
            raise

        except Exception as e:
            log.warning(
                "Bugfix JSON validation failed (attempt %d/%d): %s",
                attempt + 1, max_attempts, str(e)
            )
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Bugfix failed to produce valid FixedOutput after {max_attempts} attempts."
                ) from e

            extra_messages = [
                {"role": "assistant", "content": "I provided malformed JSON."},
                {
                    "role": "user",
                    "content": (
                        f"Your previous output failed JSON parsing/validation:\n{str(e)}\n\n"
                        f"Remember that search_text and replace_text must be valid JSON "
                        f"strings — escape all quotes and backslashes, and separate every "
                        f"field with a comma. Please try again with strict JSON compliance."
                    ),
                },
            ]

    # Apply diffs back onto the draft
    for fix in fixed.applied_fixes:
        for comp in draft.component_drafts:
            if comp.component_name == fix.component:
                comp.code = apply_diff(comp.code, fix.edits)

    state["draft_output"] = draft

    # FixedOutput has no confidence field, so call_role's internal
    # escalation only ever fires here on TruncatedOutputError, never on
    # low confidence — but bookkeeping is identical either way once it's
    # happened, so record it the same way as every other stage.
    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(fixed, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(fixed, "_escalated_from", None)
        escalated_models["bugfix"] = escalated_to_attr
        escalation_history.append({
            "stage":      "bugfix",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",
            "iteration":  iteration,
        })
        log.info("Bugfix escalated %s → %s", escalated_from_attr, escalated_to_attr)

    passed, reason = check_fixed_output_present(fixed)
    if not passed:
        log.warning("bugfix_node: output guard failed: %s", reason)

    log.info(
        "BugFix: %d applied | %d self-identified | %d unfixed | quality=%s",
        len(fixed.applied_fixes),
        len(fixed.self_identified_issues),
        len(fixed.unfixed_issues),
        fixed.overall_quality,
    )

    fixed_path = write_iteration_artifact(
        run_dir, "fixed.json", fixed.model_dump_json(indent=2), iteration,
    )

    return {
        "fixed_output":       fixed,
        "fixed_path":         fixed_path,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }