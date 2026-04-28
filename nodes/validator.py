"""
nodes/validator.py — synthesis, validation, and final cross-spec validation.

synthesise_node:     Consolidates CritiqueVerdicts → ValidationVerdict.
validate_node:       9B non-thinking gate check → routing decision.
final_validate_node: Cross-spec coherence pass on assembled sub-spec project.

Writes critique.json and verdict.json to disk.
Persists CritiqueRecord to SQLite after synthesis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.validation import (
    CritiqueRecord, CritiqueVerdict, ValidationVerdict,
    VerdictCategory, CriticScope
)
from storage.critique_store import write_critique_record

log = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_SYNTHESISE = """You are the synthesis model for a multi-critic code review ensemble.

You will receive verdicts from up to two independent critics:
  Critic A — coherence and completeness
  Critic B — logical correctness (optional)

Your job is to produce a single consolidated ValidationVerdict by:
1. Weighing each critic's verdict by domain relevance to the issues raised.
2. Escalating to the most serious verdict if critics disagree.
   (spec_problem > minor_fix > pass)
3. A Critic B constraint violation should be trusted even if Critic A passed.
4. Record in dissenting_notes any case where you overrode a critic's concern.

Return a ValidationVerdict with:
  pass         — all critics satisfied, output is good
  minor_fix    — fixable issues; include specific_issues list for the executor
  spec_problem — fundamental flaw; include description for the planner
  unresolvable — only if genuinely unresolvable after multiple iterations
"""

_SYSTEM_VALIDATE = """You are the final validation gate for a software pipeline.

You receive the FixedOutput and the original PlanSpec.
Make a clear, fast decision:

  pass         — output satisfies the plan; ready to return
  minor_fix    — specific fixable issues; list them clearly
  spec_problem — the plan itself is flawed and needs revision
  unresolvable — cannot be resolved without fundamental changes

Be decisive. Do not over-qualify. If it passes, say pass.
"""

_SYSTEM_FINAL_VALIDATE = """You are performing final validation across a multi-component project.

Multiple sub-specs have been implemented independently. Your job is to check:
1. Do the components work together coherently?
2. Are interfaces consistent — does what one component produces match what another expects?
3. Are there naming conflicts, duplicate definitions, or contradictions?
4. Is the overall project complete relative to the original task?

Flag any integration issues clearly with the components involved.
"""


# ── Synthesise ────────────────────────────────────────────────────────────────

def synthesise_node(state: PipelineState) -> dict:
    """
    Consolidate CritiqueVerdicts into a single ValidationVerdict.
    Writes critique.json. Persists CritiqueRecord to SQLite.
    """
    run_dir = state["run_dir"]
    record  = state.get("critique_record")

    if not record or not record.critic_verdicts:
        # No ensemble ran — create a pass-through record and skip to validate
        log.debug("synthesise_node: no critics ran, creating pass-through")
        placeholder = ValidationVerdict(
            category         = VerdictCategory.PASS,
            synthesis_model  = "none",
            description      = "No ensemble — passed through to gate validation",
        )
        return {"validation_verdict": placeholder}

    # Build verdicts summary for synthesis prompt
    verdicts_text = _format_verdicts(record.critic_verdicts)
    plan          = state.get("plan_spec")
    classification = state.get("classification")
    complexity     = getattr(classification, "complexity", "simple") if classification else "simple"

    # Choose synthesis model based on complexity
    role = "synthesis_complex" if complexity == "complex" else "synthesis_simple"

    start = time.perf_counter()

    messages = [
        {"role": "system", "content": _SYSTEM_SYNTHESISE},
        {
            "role": "user",
            "content": (
                f"PlanSpec summary:\n{plan.task_summary if plan else 'not available'}\n\n"
                f"Critic Verdicts:\n{verdicts_text}"
            ),
        },
    ]

    verdict: ValidationVerdict = call_role(
        role            = role,
        messages        = messages,
        response_schema = ValidationVerdict,
        stage           = "synthesise",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,  
    )

    elapsed = (time.perf_counter() - start) * 1000

    # Set synthesis model name
    verdict = verdict.model_copy(update={
        "synthesis_model": role,
        "latency_ms":      elapsed,
    })

    log.info(
        "Synthesis: %s | %d issues | dissent=%s",
        verdict.category,
        len(verdict.specific_issues),
        "yes" if verdict.dissenting_notes else "no",
    )

    # Finalise CritiqueRecord with the verdict
    record = record.model_copy(update={
        "final_verdict":       verdict,
        "ensemble_latency_ms": elapsed,
    })

    # Write critique.json to disk
    critique_path = str(Path(run_dir) / "critique.json")
    Path(critique_path).write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )

    # Persist to SQLite (Phase 1+: always store, even before retrieval is enabled)
    try:
        write_critique_record(record)
    except Exception as e:
        log.warning("Failed to persist CritiqueRecord to SQLite: %s", e)

    return {
        "critique_record":    record,
        "critique_path":      critique_path,
        "validation_verdict": verdict,
    }


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_node(state: PipelineState) -> dict:
    """
    9B non-thinking gate validation.
    If synthesis already ran, uses that verdict as context.
    Produces the final ValidationVerdict that drives routing.
    Writes verdict.json to disk.
    """
    run_dir = state["run_dir"]
    fixed   = state.get("fixed_output")
    plan    = state.get("plan_spec")
    existing_verdict = state.get("validation_verdict")

    if not fixed:
        raise ValueError("validate_node: missing fixed_output")

    # If synthesis produced a clear verdict, use it as strong context
    synthesis_ctx = ""
    if existing_verdict and existing_verdict.synthesis_model != "none":
        synthesis_ctx = (
            f"\nSynthesis verdict (from ensemble): {existing_verdict.category}\n"
            f"Description: {existing_verdict.description}\n"
        )
        if existing_verdict.specific_issues:
            synthesis_ctx += "Issues: " + "; ".join(existing_verdict.specific_issues)

    messages = [
        {"role": "system", "content": _SYSTEM_VALIDATE},
        {
            "role": "user",
            "content": (
                f"PlanSpec:\n{plan.model_dump_json(indent=2) if plan else 'not available'}"
                f"{synthesis_ctx}\n\n"
                f"FixedOutput:\n{fixed.model_dump_json(indent=2)}"
            ),
        },
    ]

    verdict: ValidationVerdict = call_role(
        role            = "validate",
        messages        = messages,
        response_schema = ValidationVerdict,
        stage           = "validate",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
    )

    # Increment iteration counter for next loop
    current_iteration = state.get("iteration", 0)
    new_iteration     = current_iteration + 1

    log.info(
        "Validation: %s | iter=%d | issues=%d",
        verdict.category, current_iteration, len(verdict.specific_issues)
    )

    # Write verdict.json
    verdict_path = str(Path(run_dir) / "verdict.json")
    Path(verdict_path).write_text(
        verdict.model_dump_json(indent=2), encoding="utf-8"
    )

    # Determine if pipeline is complete
    complete = verdict.category in (VerdictCategory.PASS, VerdictCategory.UNRESOLVABLE)
    failed   = verdict.category == VerdictCategory.UNRESOLVABLE

    return {
        "validation_verdict": verdict,
        "verdict_path":       verdict_path,
        "iteration":          new_iteration,
        "pipeline_complete":  complete,
        "pipeline_failed":    failed,
        "failure_reason":     verdict.description if failed else None,
    }


# ── Final validation (sub-spec assembly) ─────────────────────────────────────

def final_validate_node(state: PipelineState) -> dict:
    """
    Cross-spec coherence validation for decomposed tasks.
    Assembles all sub-spec outputs and checks integration.
    Writes final_validation.json.
    """
    run_dir    = state["run_dir"]
    interfaces = state.get("sub_spec_interfaces", [])
    plan       = state.get("plan_spec")

    if not interfaces:
        log.warning("final_validate_node: no sub_spec_interfaces found")
        # Write empty final output and exit
        final_path = str(Path(run_dir) / "final_validation.json")
        Path(final_path).write_text(
            json.dumps({"status": "no_sub_specs", "issues": []}), encoding="utf-8"
        )
        return {"final_validation_path": final_path, "pipeline_complete": True}

    # Run deterministic interface checks first
    from pipeline.guards import check_interface_compatibility
    compat_ok, violations = check_interface_compatibility(interfaces)

    if not compat_ok:
        log.warning("final_validate_node: %d interface violations", len(violations))
        for v in violations:
            log.warning("  Interface violation: %s", v)

    # Collect all fixed outputs from sub-spec directories
    runs_dir   = Path(run_dir)
    sub_outputs = []
    for iface in interfaces:
        sub_dir = runs_dir / "sub_specs" / iface.sub_spec_uuid
        fixed_file = sub_dir / "fixed.json"
        if fixed_file.exists():
            sub_outputs.append({
                "component": iface.component_name,
                "outputs":   iface.outputs,
                "code":      fixed_file.read_text(encoding="utf-8")[:2000],  # truncate for context
            })

    # Determine which model to use based on assembled project size
    from pathlib import Path as P
    import yaml
    cfg_path = P(__file__).parent.parent / "config" / "routing.yaml"
    with open(cfg_path) as f:
        routing = yaml.safe_load(f)
    threshold = routing.get("final_validation", {}).get(
        "escalate_to_35b_token_threshold", 12000
    )

    total_chars = sum(len(o["code"]) for o in sub_outputs)
    role = "final_validate" if total_chars / 4 < threshold else "synthesis_complex"

    messages = [
        {"role": "system", "content": _SYSTEM_FINAL_VALIDATE},
        {
            "role": "user",
            "content": (
                f"Original task summary:\n{plan.task_summary if plan else 'not available'}\n\n"
                f"Interface violations found: {len(violations)}\n"
                + (("\n".join(f"  - {v}" for v in violations) + "\n") if violations else "")
                + f"\nAssembled components ({len(sub_outputs)}):\n"
                + json.dumps(sub_outputs, indent=2)
            ),
        },
    ]

    verdict: ValidationVerdict = call_role(
        role            = role,
        messages        = messages,
        response_schema = ValidationVerdict,
        stage           = "final_validate",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
    )

    log.info(
        "Final validation: %s | compat=%s | components=%d",
        verdict.category, compat_ok, len(sub_outputs)
    )

    # Write final validation result
    final_path = str(Path(run_dir) / "final_validation.json")
    result = {
        "verdict":             verdict.model_dump(),
        "interface_compat":    compat_ok,
        "interface_violations":violations,
        "components_validated":len(sub_outputs),
    }
    Path(final_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

    complete = verdict.category in (VerdictCategory.PASS, VerdictCategory.UNRESOLVABLE)

    return {
        "final_validation_path": final_path,
        "validation_verdict":    verdict,
        "pipeline_complete":     complete,
        "pipeline_failed":       verdict.category == VerdictCategory.UNRESOLVABLE,
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_verdicts(verdicts: list[CritiqueVerdict]) -> str:
    lines = []
    for v in verdicts:
        lines.append(f"[{v.scope.upper()} — {v.critic_model}]")
        lines.append(f"  Verdict: {v.category}")
        lines.append(f"  Confidence: {v.confidence}")
        lines.append(f"  Reasoning: {v.reasoning}")
        if v.issues:
            lines.append("  Issues:")
            for issue in v.issues:
                lines.append(f"    - {issue}")
        lines.append("")
    return "\n".join(lines)
