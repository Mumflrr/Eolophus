"""
nodes/validator.py — synthesis, validation, and final cross-spec validation.

synthesise_node:     Consolidates CritiqueVerdicts → ValidationVerdict.
validate_node:       9B non-thinking gate check → routing decision.
final_validate_node: Cross-spec coherence pass on assembled sub-spec project.

Prompts live in:
  config/prompts/synthesise.yaml
  config/prompts/validate.yaml
  config/prompts/final_validate.yaml

Writes critique.json and verdict.json to disk.
Persists CritiqueRecord to SQLite after synthesis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from clients.llm import call_role, write_iteration_artifact
from pipeline.state import PipelineState
from schemas.validation import (
    CritiqueRecord, CritiqueVerdict, ValidationVerdict,
    VerdictCategory, CriticScope
)
from storage.critique_store import write_critique_record, update_critique_resolved

log = logging.getLogger(__name__)


# ── Synthesise ────────────────────────────────────────────────────────────────

def synthesise_node(state: PipelineState) -> dict:
    """Consolidate CritiqueVerdicts into a single ValidationVerdict."""
    run_dir   = state["run_dir"]
    record    = state.get("critique_record")
    iteration = state.get("iteration", 0)

    if not record or not record.critic_verdicts:
        log.debug("synthesise_node: no critics ran — pass-through")
        placeholder = ValidationVerdict(
            category        = VerdictCategory.PASS,
            synthesis_model = "none",
            description     = "No ensemble — passed through to gate validation",
        )
        return {"validation_verdict": placeholder}

    verdicts_text  = _format_verdicts(record.critic_verdicts)
    plan           = state.get("plan_spec")
    classification = state.get("classification")
    complexity     = getattr(classification, "complexity", "simple") if classification else "simple"
    role           = "synthesis_complex" if complexity == "complex" else "synthesis_simple"

    start = time.perf_counter()

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get(role)

    verdict: ValidationVerdict = call_role(
        role            = role,
        template_vars   = {
            "task_summary": plan.task_summary if plan else "not available",
            "verdicts_text":verdicts_text,
        },
        response_schema = ValidationVerdict,
        stage           = "synthesise",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(verdict, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(verdict, "_escalated_from", None)
        escalated_models[role] = escalated_to_attr
        escalation_history.append({
            "stage":      role,
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",   # ValidationVerdict has no confidence field
            "iteration":  iteration,
        })
        log.info("Synthesise (%s) escalated %s → %s", role, escalated_from_attr, escalated_to_attr)

    elapsed = (time.perf_counter() - start) * 1000
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

    record = record.model_copy(update={
        "final_verdict":       verdict,
        "ensemble_latency_ms": elapsed,
        "resolved":            verdict.category == VerdictCategory.PASS,
    })

    critique_path = write_iteration_artifact(
        run_dir, "critique.json", record.model_dump_json(indent=2), iteration,
    )

    try:
        write_critique_record(record)
    except Exception as e:
        log.warning("Failed to persist CritiqueRecord: %s", e)

    return {
        "critique_record":    record,
        "critique_path":      critique_path,
        "validation_verdict": verdict,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_node(state: PipelineState) -> dict:
    """9B non-thinking gate validation. Writes verdict.json."""
    run_dir          = state["run_dir"]
    fixed            = state.get("fixed_output")
    plan             = state.get("plan_spec")
    existing_verdict = state.get("validation_verdict")

    if not fixed:
        raise ValueError("validate_node: missing fixed_output")

    synthesis_context = ""
    if existing_verdict and existing_verdict.synthesis_model != "none":
        synthesis_context = (
            f"Synthesis verdict (from ensemble): {existing_verdict.category}\n"
            f"Description: {existing_verdict.description}\n"
        )
        if existing_verdict.specific_issues:
            synthesis_context += "Issues: " + "; ".join(existing_verdict.specific_issues)

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("validate")

    verdict: ValidationVerdict = call_role(
        role            = "validate",
        template_vars   = {
            "plan_json":         plan.model_dump_json(indent=2) if plan else "not available",
            "synthesis_context": synthesis_context,
            "fixed_json":        fixed.model_dump_json(indent=2),
        },
        response_schema = ValidationVerdict,
        stage           = "validate",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(verdict, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(verdict, "_escalated_from", None)
        escalated_models["validate"] = escalated_to_attr
        escalation_history.append({
            "stage":      "validate",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Validate escalated %s → %s", escalated_from_attr, escalated_to_attr)

    current_iteration = state.get("iteration", 0)
    new_iteration     = current_iteration + 1

    log.info(
        "Validation: %s | iter=%d | issues=%d",
        verdict.category, current_iteration, len(verdict.specific_issues)
    )

    # Update SQLite resolved flag — synthesise_node wrote a provisional value;
    # validate_node has the final word and corrects it here.
    critique_record = state.get("critique_record")
    if critique_record:
        is_resolved = verdict.category == VerdictCategory.PASS
        try:
            update_critique_resolved(state["run_uuid"], current_iteration, is_resolved)
        except Exception as e:
            log.warning("Failed to update critique resolved flag: %s", e)

    verdict_path = write_iteration_artifact(
        run_dir, "verdict.json", verdict.model_dump_json(indent=2), current_iteration,
    )

    complete = verdict.category in (VerdictCategory.PASS, VerdictCategory.UNRESOLVABLE)
    failed   = verdict.category == VerdictCategory.UNRESOLVABLE

    return {
        "validation_verdict": verdict,
        "verdict_path":       verdict_path,
        "iteration":          new_iteration,
        "pipeline_complete":  complete,
        "pipeline_failed":    failed,
        "failure_reason":     verdict.description if failed else None,
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }


# ── Final validation ──────────────────────────────────────────────────────────

def final_validate_node(state: PipelineState) -> dict:
    """Cross-spec coherence validation for decomposed tasks."""
    run_dir    = state["run_dir"]
    interfaces = state.get("sub_spec_interfaces", [])
    plan       = state.get("plan_spec")

    if not interfaces:
        log.warning("final_validate_node: no sub_spec_interfaces found")
        final_path = str(Path(run_dir) / "final_validation.json")
        Path(final_path).write_text(
            json.dumps({"status": "no_sub_specs", "issues": []}),
            encoding="utf-8"
        )
        return {"final_validation_path": final_path, "pipeline_complete": True}

    # Deterministic interface compatibility check
    from pipeline.guards import check_interface_compatibility
    compat_ok, violations = check_interface_compatibility(interfaces)

    if not compat_ok:
        log.warning("final_validate_node: %d interface violations", len(violations))

    # Collect sub-spec outputs (truncated for context)
    runs_dir    = Path(run_dir)
    sub_outputs = []
    for iface in interfaces:
        sub_dir    = runs_dir / "sub_specs" / iface.sub_spec_uuid
        fixed_file = sub_dir / "fixed.json"
        if fixed_file.exists():
            sub_outputs.append({
                "component": iface.component_name,
                "outputs":   iface.outputs,
                "code":      fixed_file.read_text(encoding="utf-8")[:2000],
            })

    # Escalate to 35B if assembled project is large
    import yaml
    cfg_path  = Path(__file__).parent.parent / "config" / "routing.yaml"
    with open(cfg_path) as f:
        routing = yaml.safe_load(f)
    threshold = routing.get("final_validation", {}).get(
        "escalate_to_35b_token_threshold", 12000
    )
    total_chars = sum(len(o["code"]) for o in sub_outputs)
    role = "final_validate" if total_chars / 4 < threshold else "synthesis_complex"

    violations_block = (
        "\n".join(f"  - {v}" for v in violations) + "\n"
        if violations else ""
    )

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get(role)

    verdict: ValidationVerdict = call_role(
        role            = role,
        template_vars   = {
            "task_summary":     plan.task_summary if plan else "not available",
            "violation_count":  str(len(violations)),
            "violations_block": violations_block,
            "component_count":  str(len(sub_outputs)),
            "sub_outputs_json": json.dumps(sub_outputs, indent=2),
        },
        response_schema = ValidationVerdict,
        stage           = "final_validate",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(verdict, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(verdict, "_escalated_from", None)
        escalated_models[role] = escalated_to_attr
        escalation_history.append({
            "stage":      role,
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Final validate (%s) escalated %s → %s", role, escalated_from_attr, escalated_to_attr)

    log.info(
        "Final validation: %s | compat=%s | components=%d",
        verdict.category, compat_ok, len(sub_outputs)
    )

    final_path = str(Path(run_dir) / "final_validation.json")
    Path(final_path).write_text(
        json.dumps({
            "verdict":              verdict.model_dump(),
            "interface_compat":     compat_ok,
            "interface_violations": violations,
            "components_validated": len(sub_outputs),
        }, indent=2),
        encoding="utf-8"
    )

    complete = verdict.category in (VerdictCategory.PASS, VerdictCategory.UNRESOLVABLE)

    return {
        "final_validation_path": final_path,
        "validation_verdict":    verdict,
        "pipeline_complete":     complete,
        "pipeline_failed":       verdict.category == VerdictCategory.UNRESOLVABLE,
        "escalated_models":      escalated_models,
        "escalation_history":    escalation_history,
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