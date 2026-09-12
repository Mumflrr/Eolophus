"""
nodes/finalize.py — assembles the user-facing final answer.

Nothing else in this graph writes final.json / sets final_output_path.
distiller_node (nodes/distiller.py) is the terminal node before END, but
its entire job is self-improvement bookkeeping (extracting a lesson into
the lesson store when the run iterated before passing) — it never builds
or writes a final answer. Every successful run therefore reached END with
final_output_path unset, and server.py's _extract_reply_text() fell
through its fallback chain to `fixed_output`/`draft_output`, returning
str(a_pydantic_model) — a raw repr like "applied_fixes=[AppliedFix(...)]
self_identified_issues=[...] overall_quality='...'" — as the chat reply.
That's not a rare failure case; it happened on every run that reached
distiller successfully, since final_output_path was structurally never
populated on that path.

This node sits between validate (on a "pass" verdict) and distiller,
building a plain-text answer from the final draft/fixed code and writing
it to final.json as {"answer": ...} so _extract_reply_text has something
real to read instead of falling back to a raw object repr.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.state import PipelineState

log = logging.getLogger(__name__)


def finalize_node(state: PipelineState) -> dict:
    """
    Build the final user-facing answer from draft_output (which already
    has bugfixer.py's fixes applied in-place — see bugfix_node's
    `state["draft_output"] = draft` after apply_diff) and write final.json.
    """
    run_dir = state["run_dir"]
    draft   = state.get("draft_output")
    plan    = state.get("plan_spec")

    if not draft:
        # Nothing to assemble from — leave final_output_path unset rather
        # than writing a misleading empty answer; _extract_reply_text's
        # failure_reason fallback will surface something sensible instead.
        log.warning("finalize_node: no draft_output in state — skipping")
        return {}

    parts = []
    if plan and plan.task_summary:
        parts.append(f"# {plan.task_summary}\n")

    for comp in draft.component_drafts:
        parts.append(f"## {comp.component_name}\n")
        parts.append(f"```python\n{comp.code}\n```\n")
        if comp.notes:
            parts.append(f"_Note: {comp.notes}_\n")

    if draft.implementation_notes:
        parts.append(f"\n**Implementation notes:** {draft.implementation_notes}")

    answer = "\n".join(parts).strip()

    final_path = str(Path(run_dir) / "final.json")
    Path(final_path).write_text(
        json.dumps({"answer": answer}, indent=2),
        encoding="utf-8",
    )

    log.info("Finalize: wrote final.json (%d components, %d chars)",
              len(draft.component_drafts), len(answer))

    return {
        "final_output_path": final_path,
    }