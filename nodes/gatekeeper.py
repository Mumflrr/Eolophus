"""
nodes/gatekeeper.py — breakpoint and safety node.

AI-driven stop logic uses the dedicated 'gatekeeper' role (9B, non-thinking).
Human-in-the-loop pause uses LangGraph interrupt().

Prompt lives in config/prompts/gatekeeper.yaml.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt
from clients.llm import call_role
from pipeline.state import PipelineState
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class AuditDecision(BaseModel):
    decision: str = Field(description="Must be 'CONTINUE' or 'HALT'")
    reason:   str = Field(description="Explanation for the decision")


def gatekeeper_node(state: PipelineState) -> dict:
    """Evaluates if the pipeline should pause for human review or halt."""
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

        if audit.decision == "HALT":
            log.warning("AI-driven stop triggered: %s", audit.reason)
            return {"_halt_reason": audit.reason, "status": "unresolvable"}

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
