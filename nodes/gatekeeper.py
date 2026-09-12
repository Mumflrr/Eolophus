"""
nodes/gatekeeper.py — breakpoint and safety node.

AI-driven stop logic uses the dedicated 'gatekeeper' role (9B, non-thinking).
Human-in-the-loop pause uses LangGraph interrupt().

Prompt lives in config/prompts/gatekeeper.yaml.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt
from clients.llm import call_role, TruncatedOutputError, write_iteration_artifact
from pipeline.state import PipelineState
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class AuditDecision(BaseModel):
    decision: str = Field(description="Must be 'CONTINUE' or 'HALT'")
    reason:   str = Field(description="Explanation for the decision")


def gatekeeper_node(state: PipelineState) -> dict:
    """Evaluates if the pipeline should pause for human review or halt.

    Writes audit.json (via write_iteration_artifact, same as
    bugfixer/validator/drafter) whenever the AI-driven stop call actually
    succeeds — previously this node wrote NOTHING to disk despite the
    'audit' stage showing real token counts and retries in the stages
    timeline: the AuditDecision (decision + reason) only ever went into a
    log line (on HALT) or straight into in-memory state, with no artifact
    a person could open the way fixed.json/verdict.json/critique.json can
    be. Skipped entirely on the except-Exception fallback below, since in
    that branch no AuditDecision was ever produced — there's nothing
    to write, and writing a placeholder would misrepresent a failed call
    as a real audit.
    """
    run_dir   = state.get("run_dir", "")
    iteration = state.get("iteration", 0)
    verdict   = state.get("validation_verdict")
    verdict_str = str(getattr(verdict, "specific_issues", "None")) if verdict else "None"

    # ── 1. AI-driven stop ─────────────────────────────────────────────────
    try:
        audit: AuditDecision = call_role(
            role            = "gatekeeper",
            template_vars   = {
                "iteration":          str(iteration),
                "validation_verdict": verdict_str,
            },
            response_schema = AuditDecision,
            stage           = "audit",
            run_dir         = run_dir,
            thinking        = False,
        )

        write_iteration_artifact(
            run_dir, "audit.json", audit.model_dump_json(indent=2), iteration,
        )

        if audit.decision == "HALT":
            log.warning("AI-driven stop triggered: %s", audit.reason)
            return {"_halt_reason": audit.reason, "status": "unresolvable"}

    except TruncatedOutputError:
        # Let this propagate — previously caught by the bare `except
        # Exception` below and silently downgraded to "defaulting to
        # CONTINUE", which meant a truncated gatekeeper call was the one
        # place in the whole pipeline where truncation was invisible even
        # in the logs' effect (the run just continued as if gatekeeper had
        # said CONTINUE on purpose). Re-raising lets pipeline/graph.py's
        # generic _wrap_node_for_truncation_retry catch it at the node-
        # wrapper level instead, same as every other node, so it can be
        # surfaced and retried with a higher cap rather than swallowed.
        raise
    except Exception as e:
        log.warning("Gatekeeper audit failed, defaulting to CONTINUE: %s", e)

    # ── 2. Human-driven stop ──────────────────────────────────────────────
    if state.get("force_human_review") or iteration >= 3:
        log.info("Graph paused for human review at iteration %d.", iteration)
        user_input = interrupt({"question": "Pipeline paused. Continue? (yes/no)"})

        if user_input.lower() not in ["y", "yes"]:
            log.warning("Human aborted the run.")
            return {"_halt_reason": "Human aborted.", "status": "interrupted"}

    return {"_gate_passed": True}