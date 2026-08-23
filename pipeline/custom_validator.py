"""
pipeline/custom_validator.py — structural validation for PipelineDefinition,
beyond what Pydantic's field-level validators already catch.

Pydantic validation (in schemas/pipeline_def.py) catches per-field problems:
duplicate ids, missing max_iterations on a self-declared loop-back, bad
entry_step. This module catches GRAPH-SHAPE problems that require looking
at the whole step list together — the kind of thing you can only find by
tracing edges.

Called from:
  - pipeline/custom_graph.py, before compiling (refuses to compile invalid defs)
  - api/server.py's pipeline CRUD endpoints, before saving (refuses to save
    invalid defs, so the GUI gets an error at edit time, not at run time)

Returns a list of human-readable error strings. Empty list = valid.
Deliberately returns strings rather than raising, so callers can decide
whether to reject entirely or show warnings — right now both callers
treat any non-empty list as a hard rejection, but that's a caller choice,
not something baked into this module.
"""

from __future__ import annotations

from schemas.pipeline_def import PipelineDefinition, StepType, DecisionStep


def validate_pipeline_definition(definition: PipelineDefinition) -> list[str]:
    errors: list[str] = []

    step_ids = {s.id for s in definition.steps}
    step_by_id = {s.id: s for s in definition.steps}

    errors.extend(_check_edge_targets_exist(definition, step_ids))
    errors.extend(_check_undeclared_cycles(definition, step_by_id))
    errors.extend(_check_reachability(definition, step_by_id))
    errors.extend(_check_freeform_input_keys(definition, step_by_id))
    errors.extend(_check_decision_outcomes_exhaustive(definition))

    return errors


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_edge_targets_exist(definition: PipelineDefinition, step_ids: set[str]) -> list[str]:
    """Every next_step / edge_override target must be a real step id or __end__."""
    errors = []

    for step in definition.steps:
        if step.type == StepType.DECISION:
            for outcome in step.outcomes:
                if outcome.next_step != "__end__" and outcome.next_step not in step_ids:
                    errors.append(
                        f"Decision step '{step.id}' outcome '{outcome.value}' routes to "
                        f"'{outcome.next_step}', which is not a defined step id"
                    )

    for from_id, to_id in definition.edge_overrides.items():
        if from_id not in step_ids:
            errors.append(f"edge_override references unknown source step '{from_id}'")
        if to_id != "__end__" and to_id not in step_ids:
            errors.append(f"edge_override for '{from_id}' targets unknown step '{to_id}'")

    return errors


def _check_undeclared_cycles(
    definition: PipelineDefinition, step_by_id: dict
) -> list[str]:
    """
    Every edge that points BACKWARD in step-list order (or to itself) from a
    decision node must have is_loop_back=True set. This is the check that
    stops someone from wiring a cycle and forgetting to cap it — Pydantic
    already ensures is_loop_back=True implies max_iterations is set, but
    nothing stops someone from wiring a cycle WITHOUT setting is_loop_back
    in the first place. This check closes that gap.

    Only decision nodes can create cycles (linear/edge_override edges only
    ever point at "next in list" or an explicit override — an override
    pointing backward is also flagged here since it's an uncapped cycle
    with no mechanism to break it at all).
    """
    errors = []
    step_order = [s.id for s in definition.steps]
    index_of = {sid: i for i, sid in enumerate(step_order)}

    for step in definition.steps:
        if step.type == StepType.DECISION:
            for outcome in step.outcomes:
                if outcome.next_step == "__end__":
                    continue
                target_idx = index_of.get(outcome.next_step)
                self_idx   = index_of[step.id]
                is_backward = target_idx is not None and target_idx <= self_idx
                if is_backward and not step.is_loop_back:
                    errors.append(
                        f"Decision step '{step.id}' outcome '{outcome.value}' routes "
                        f"backward to '{outcome.next_step}' (creates a cycle) but "
                        f"is_loop_back is not set to True. Every cycle must be "
                        f"explicitly marked so max_iterations is enforced."
                    )

    for from_id, to_id in definition.edge_overrides.items():
        if to_id == "__end__":
            continue
        from_idx = index_of.get(from_id)
        to_idx   = index_of.get(to_id)
        if from_idx is not None and to_idx is not None and to_idx <= from_idx:
            errors.append(
                f"edge_override '{from_id}' -> '{to_id}' points backward, creating an "
                f"UNCAPPED cycle. Only decision steps (with is_loop_back=True and "
                f"max_iterations) may create cycles. Route through a decision step instead."
            )

    return errors


def _check_reachability(definition: PipelineDefinition, step_by_id: dict) -> list[str]:
    """Every declared step should be reachable from entry_step. Unreachable
    steps are almost certainly a mistake (a typo'd edge, a leftover from
    an earlier edit) rather than intentional dead code."""
    errors = []
    step_order = [s.id for s in definition.steps]
    index_of   = {sid: i for i, sid in enumerate(step_order)}

    visited: set[str] = set()
    stack = [definition.entry_step]
    while stack:
        current = stack.pop()
        if current in visited or current not in step_by_id:
            continue
        visited.add(current)

        step = step_by_id[current]
        if step.type == StepType.DECISION:
            for outcome in step.outcomes:
                if outcome.next_step != "__end__":
                    stack.append(outcome.next_step)
        else:
            override = definition.edge_overrides.get(current)
            if override and override != "__end__":
                stack.append(override)
            elif not override:
                idx = index_of[current]
                if idx + 1 < len(step_order):
                    stack.append(step_order[idx + 1])

    unreached = set(step_by_id.keys()) - visited
    for step_id in sorted(unreached):
        errors.append(
            f"Step '{step_id}' is unreachable from entry_step "
            f"'{definition.entry_step}' — check for a typo in an edge target"
        )

    return errors


def _check_freeform_input_keys(definition: PipelineDefinition, step_by_id: dict) -> list[str]:
    """
    Warn (as an error, for now — could be downgraded to warning-only later)
    when a freeform/decision step's input_key doesn't correspond to ANY
    prior step's output and isn't one of the pipeline's built-in initial
    state keys. This mostly catches typos ('draft_ouput' instead of
    'draft_output') that would otherwise silently produce an empty prompt
    at run time instead of failing loudly at save time.
    """
    from schemas.pipeline_def import ExistingStep, FreeformStep

    # Known output keys each existing node writes (mirrors what those node
    # functions actually return — kept here rather than imported from the
    # nodes/ package to avoid a heavy import just for a key-name check).
    EXISTING_OUTPUT_KEYS = {
        "classify":      {"classification", "mode", "task_type"},
        "vision_decode": {"visual_description", "normalised_input"},
        "ideation":      {"ideation_output"},
        "plan":          {"plan_spec"},
        "draft":         {"draft_output"},
        "draft_short":   {"draft_output"},
        "appraise":      {"appraisal_report"},
        "bugfix":        {"fixed_output"},
        "critic_a":      {"critique_record"},
        "critic_b":      {"critique_record"},
        "synthesise":    {"validation_verdict"},
        "validate":      {"validation_verdict"},
        "describe":      {"final_output_path"},
        "distiller":     set(),
    }
    INITIAL_STATE_KEYS = {"raw_text_input", "normalised_input", "raw_image_path"}

    available: set[str] = set(INITIAL_STATE_KEYS)
    errors = []

    for step in definition.steps:
        if step.type in (StepType.FREEFORM, StepType.DECISION):
            if step.input_key not in available:
                errors.append(
                    f"Step '{step.id}' reads input_key '{step.input_key}', which no "
                    f"prior step produces (available at this point: {sorted(available)}). "
                    f"Likely a typo, or the step is positioned before its data exists."
                )

        if isinstance(step, ExistingStep):
            available |= EXISTING_OUTPUT_KEYS.get(step.node_name, set())
        elif isinstance(step, FreeformStep):
            available.add(step.output_key)

    return errors


def _check_decision_outcomes_exhaustive(definition: PipelineDefinition) -> list[str]:
    """Every decision step needs at least 2 distinct outcome values —
    Pydantic's min_length=2 already covers count, this covers duplicates."""
    errors = []
    for step in definition.steps:
        if step.type == StepType.DECISION:
            values = [o.value for o in step.outcomes]
            if len(values) != len(set(values)):
                errors.append(
                    f"Decision step '{step.id}' has duplicate outcome values: {values}"
                )
    return errors
