"""
nodes/gatekeeper.py — Breakpoint and safety node.
"""

import logging
from langgraph.types import interrupt
from clients.llm import call_role
from pipeline.state import PipelineState
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

class AuditDecision(BaseModel):
    decision: str = Field(description="Must be 'CONTINUE' or 'HALT'")
    reason: str = Field(description="Explanation for the decision")

def gatekeeper_node(state: PipelineState) -> dict:
    """
    Evaluates if the pipeline should pause for human review or halt entirely.
    """
    run_dir = state.get("run_dir", "")
    
    # 1. AI-DRIVEN STOP LOGIC (Self-Correction)
    # We ask the 9B model: "Is the current progress valid, or are we looping?"
    messages = [
        {"role": "system", "content": "You are a pipeline auditor. Review the current state. If the pipeline is stuck in a loop or producing garbage, output HALT. Otherwise output CONTINUE."},
        {"role": "user", "content": f"Iteration: {state.get('iteration', 0)}. Recent feedback: {state.get('validation_verdict', 'None')}"}
    ]
    
    try:
        audit = call_role(
            role="classify", # Reusing 9B for fast audit
            messages=messages,
            response_schema=AuditDecision,
            stage="audit",
            run_dir=run_dir,
            thinking=False
        )
        
        if audit.decision == "HALT":
            log.warning("AI-Driven Stop triggered: %s", audit.reason)
            return {"_halt_reason": audit.reason, "status": "unresolvable"}
            
    except Exception as e:
        log.warning("Audit failed, defaulting to continue: %s", e)

    # 2. HUMAN-DRIVEN STOP LOGIC
    # Pauses the graph and yields control back to run.py
    if state.get("force_human_review") or state.get("iteration", 0) >= 3:
        log.info("Graph paused for human review.")
        # The graph physically stops here and returns this payload to run.py
        user_input = interrupt({"question": "Pipeline paused. Continue to execution? (yes/no)"})
        
        if user_input.lower() not in ["y", "yes"]:
            log.warning("Human aborted the run.")
            return {"_halt_reason": "Human aborted.", "status": "interrupted"}

    return {"_gate_passed": True}