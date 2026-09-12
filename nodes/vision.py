"""
nodes/vision.py — 9B vision decode.

Converts image input to VisualDescription Pydantic object,
then normalises it into a text task schema for downstream nodes.
After this node, all downstream models see text only.

System prompt lives in config/prompts/vision.yaml.
The user message contains structured image_url content (not plain text),
so Mode B (explicit messages) is used — system from YAML, user built here.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from clients.llm import call_role, load_prompt
from pipeline.state import PipelineState
from schemas.visual_description import VisualDescription

log = logging.getLogger(__name__)


def vision_decode_node(state: PipelineState) -> dict:
    """
    Decode an image input into a VisualDescription.
    Merges the description with any accompanying text input.
    """
    run_dir    = state["run_dir"]
    image_path = state.get("raw_image_path", "")
    text_input = state.get("raw_text_input", "")

    if not image_path or not Path(image_path).exists():
        log.warning("vision_decode_node: no valid image path, skipping")
        return {"normalised_input": text_input}

    # ── Encode image ───────────────────────────────────────────────────────
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    # ── Build messages — Mode B (system from YAML, user is structured) ────
    prompt_def  = load_prompt("vision_decode")
    system_text = prompt_def.get("system", (
        "Analyse this image and return a structured VisualDescription."
    ))

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {
                    "type":      "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"Additional context from user: {text_input}"
                        if text_input else
                        "Please analyse this image and return a structured VisualDescription."
                    ),
                },
            ],
        },
    ]

    profile = state.get("profile") or state.get("requested_profile")
    current_model_override = (state.get("escalated_models") or {}).get("vision_decode")

    description: VisualDescription = call_role(
        role            = "vision_decode",
        messages        = messages,
        response_schema = VisualDescription,
        stage           = "vision",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,
        profile         = profile,
        current_model_override = current_model_override,
    )

    escalated_models   = dict(state.get("escalated_models") or {})
    escalation_history = list(state.get("escalation_history") or [])
    escalated_to_attr  = getattr(description, "_escalated_to", None)
    if escalated_to_attr:
        escalated_from_attr = getattr(description, "_escalated_from", None)
        escalated_models["vision_decode"] = escalated_to_attr
        escalation_history.append({
            "stage":      "vision_decode",
            "from_model": escalated_from_attr,
            "to_model":   escalated_to_attr,
            "trigger":    "low_confidence" if description.confidence != "low" else "truncation",
            "iteration":  state.get("iteration", 0),
        })
        log.info("Vision decode escalated %s → %s", escalated_from_attr, escalated_to_attr)

    human_in_the_loop = state.get("human_in_the_loop", True)
    if description.confidence == "low" and description.clarification_question:
        if human_in_the_loop:
            log.warning(
                "Vision decode halted — needs human input: %s",
                description.clarification_question
            )
            return {
                "pipeline_halted":      True,
                "clarification_needed": description.clarification_question,
                "escalated_models":     escalated_models,
                "escalation_history":   escalation_history,
            }
        log.warning(
            "Vision decode confidence=low after escalation exhausted, but "
            "human_in_the_loop=False (set-and-forget) — proceeding "
            "best-effort. Original question was: %s",
            description.clarification_question,
        )

    log.info(
        "Vision decode: %s — %d elements | %d requirements",
        description.content_type,
        len(description.ui_elements),
        len(description.inferred_requirements),
    )

    # ── Normalise to text task description ────────────────────────────────
    parts = [
        f"[Visual Input: {description.content_type}]",
        description.summary,
        "",
        description.structural_description,
    ]
    if description.inferred_requirements:
        parts.append("\nInferred requirements:")
        parts.extend(f"  - {r}" for r in description.inferred_requirements)
    if description.ambiguities:
        parts.append("\nAmbiguities to resolve during planning:")
        parts.extend(f"  - {a}" for a in description.ambiguities)
    if text_input:
        parts.append(f"\nUser instruction: {text_input}")

    return {
        "visual_description": description,
        "normalised_input":   "\n".join(parts),
        "escalated_models":   escalated_models,
        "escalation_history": escalation_history,
    }