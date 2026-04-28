"""
nodes/vision.py — 9B vision decode.

Converts image input to VisualDescription Pydantic object,
then normalises it into a text task schema for downstream nodes.
After this node, all downstream models see text only.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from clients.llm import call_role
from pipeline.state import PipelineState
from schemas.visual_description import VisualDescription

log = logging.getLogger(__name__)

_SYSTEM = """You are analysing a visual input (image) to extract structured information
for a software development pipeline. Describe everything you can see precisely and completely.
Pay special attention to UI elements, data flows, component relationships, and any text visible in the image.

Reflect your confidence in the `confidence` field. If the image is blurry, cut off, or you are unsure what it depicts, set confidence to 'low' and write a specific question to the user in `clarification_question`."""


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

    # Encode image to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine image MIME type from extension
    ext = Path(image_path).suffix.lower()
    ext = ext[1:] if ext.startswith(".") else "png"
    if ext == "jpg":
        ext = "jpeg"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
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

    description: VisualDescription = call_role(
        role            = "vision_decode",
        messages        = messages,
        response_schema = VisualDescription,
        stage           = "vision",
        run_dir         = run_dir,
        thinking        = False,
        max_retries     = 0,      # <--- Added timeout protection
    )

    # <--- Added Human-in-the-Loop Halting Logic
    if description.confidence == "low" and description.clarification_question:
        log.warning("Vision decode halted — needs human input: %s", description.clarification_question)
        return {
            "pipeline_halted": True,
            "clarification_needed": description.clarification_question
        }

    log.info("Vision decode: %s — %d elements, %d requirements",
             description.content_type,
             len(description.ui_elements),
             len(description.inferred_requirements))

    # Normalise into a task description string
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

    normalised = "\n".join(parts)

    return {
        "visual_description": description,
        "normalised_input":   normalised,
    }