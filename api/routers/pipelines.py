"""
api/routers/pipelines.py — CRUD for custom pipeline definitions
(config/pipelines/{name}.yaml).

Every write goes through the same validate_pipeline_definition() the
compiler itself uses (pipeline/custom_validator.py), so a definition
that fails to save would also have failed to compile — the GUI never
gets a false "saved OK" that then blows up at run time.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from fastapi import APIRouter, HTTPException

from api.schemas import PipelineDefIn
if TYPE_CHECKING:
    from schemas.pipeline_def import PipelineDefinition
from schemas.pipeline_def import FeedbackMode
router = APIRouter()


def _wire_to_definition(body: PipelineDefIn) -> PipelineDefinition:
    """Convert the API's flat wire format into the real tagged-union schema."""
    from schemas.pipeline_def import (
        PipelineDefinition, ExistingStep, FreeformStep, DecisionStep,
        DecisionOutcome, StepType,
    )

    steps = []
    for s in body.steps:
        if s.type == "existing":
            if not s.node_name:
                raise HTTPException(400, f"Step '{s.id}': type=existing requires node_name")
            steps.append(ExistingStep(
                id=s.id, node_name=s.node_name,
                model_override=s.model_override, budget_override=s.budget_override,
            ))
        elif s.type == "freeform":
            if not s.model or not s.system_prompt or not s.output_key:
                raise HTTPException(
                    400, f"Step '{s.id}': type=freeform requires model, system_prompt, output_key"
                )
            steps.append(FreeformStep(
                id=s.id, model=s.model,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                thinking=s.thinking or False,
                system_prompt=s.system_prompt,
                user_template=s.user_template or "{input}",
                input_key=s.input_key or "normalised_input",
                output_key=s.output_key,
                feedback_mode=FeedbackMode(s.feedback_mode or "auto"),
            ))
        elif s.type == "decision":
            if not s.system_prompt or not s.outcomes:
                raise HTTPException(
                    400, f"Step '{s.id}': type=decision requires system_prompt, outcomes"
                )
            steps.append(DecisionStep(
                id=s.id, model=s.model or "9b",
                thinking=s.thinking or False,
                budget_tokens=s.budget_tokens if s.budget_tokens is not None else 0,
                system_prompt=s.system_prompt,
                input_key=s.input_key or "normalised_input",
                outcomes=[DecisionOutcome(**o) for o in s.outcomes],
                is_loop_back=s.is_loop_back or False,
                max_iterations=s.max_iterations,
                feedback_mode=FeedbackMode(s.feedback_mode or "auto"),
            ))
        else:
            raise HTTPException(400, f"Step '{s.id}': unknown type '{s.type}'")

    return PipelineDefinition(
        name=body.name, description=body.description,
        entry_step=body.entry_step, steps=steps,
        edge_overrides=body.edge_overrides,
        max_total_iterations=body.max_total_iterations,
    )


@router.get("/pipelines")
async def list_pipelines():
    """List all saved custom pipeline definitions (name + description only)."""
    from pipeline.custom_graph import PIPELINES_DIR, load_pipeline_definition

    result = []
    for path in sorted(PIPELINES_DIR.glob("*.yaml")):
        try:
            defn = load_pipeline_definition(path.stem)
            result.append({
                "name": defn.name, "description": defn.description,
                "step_count": len(defn.steps), "entry_step": defn.entry_step,
            })
        except Exception as e:
            result.append({"name": path.stem, "description": f"[INVALID: {e}]", "step_count": 0})
    return result


@router.get("/pipelines/{name}")
async def get_pipeline(name: str):
    """Full definition for one custom pipeline — the shape a GUI editor needs."""
    from pipeline.custom_graph import load_pipeline_definition
    try:
        defn = load_pipeline_definition(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    return json.loads(defn.model_dump_json())


@router.post("/pipelines/validate")
async def validate_pipeline(body: PipelineDefIn):
    """
    Validate a pipeline definition WITHOUT saving it. Lets a GUI show
    live errors while the user is still editing, before they commit.
    """
    from pipeline.custom_validator import validate_pipeline_definition
    try:
        defn = _wire_to_definition(body)   # raises HTTPException on structural issues
    except HTTPException:
        raise
    except Exception as e:
        # Pydantic validation errors (duplicate ids, missing max_iterations
        # on a self-declared loop-back, bad entry_step, etc.)
        return {"valid": False, "errors": [str(e)]}

    errors = validate_pipeline_definition(defn)
    return {"valid": len(errors) == 0, "errors": errors}


@router.post("/pipelines")
async def create_pipeline(body: PipelineDefIn):
    """
    Create or overwrite a custom pipeline definition. Validates before
    writing — a pipeline that fails validation is never saved to disk,
    so config/pipelines/ never contains a definition that would fail to
    compile at run time.
    """
    from pipeline.custom_graph import PIPELINES_DIR
    from pipeline.custom_validator import validate_pipeline_definition

    defn = _wire_to_definition(body)
    errors = validate_pipeline_definition(defn)
    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Pipeline failed validation, not saved", "errors": errors},
        )

    path = PIPELINES_DIR / f"{body.name}.yaml"
    import yaml
    with open(path, "w") as f:
        yaml.dump(json.loads(defn.model_dump_json()), f, default_flow_style=False, sort_keys=False)

    return {"status": "saved", "name": body.name, "path": str(path)}


@router.delete("/pipelines/{name}")
async def delete_pipeline(name: str):
    """Delete a custom pipeline definition. Does not affect past runs."""
    from pipeline.custom_graph import PIPELINES_DIR
    path = PIPELINES_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No pipeline named '{name}'")
    path.unlink()
    return {"status": "deleted", "name": name}


@router.get("/pipelines/nodes/available")
async def list_available_node_types():
    """
    What a GUI needs to populate an 'add step' dropdown: every reusable
    existing-node name, and the step-type shapes (freeform/decision)
    with their required fields, sourced directly from the real schema
    so this never drifts out of sync with what the backend actually accepts.
    """
    from schemas.pipeline_def import EXISTING_NODE_NAMES

    return {
        "existing_nodes": EXISTING_NODE_NAMES,
        "step_types": {
            "existing": {
                "required": ["id", "node_name"],
                "optional": ["model_override", "budget_override"],
            },
            "freeform": {
                "required": ["id", "model", "system_prompt", "output_key"],
                "optional": ["budget_tokens", "thinking", "user_template",
                             "input_key", "feedback_mode"],
            },
            "decision": {
                "required": ["id", "system_prompt", "outcomes"],
                "optional": ["model", "thinking", "budget_tokens", "input_key",
                             "is_loop_back", "max_iterations", "feedback_mode"],
                "outcome_shape": {"value": "string", "next_step": "string (step id or __end__)",
                                   "description": "string, optional"},
            },
        },
    }
