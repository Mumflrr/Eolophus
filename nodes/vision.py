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
Pay special attention to UI elements, data flows, component relationships, and any text visible in the image."""


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
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Please analyse this image and return a structured VisualDescription.\n"
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
    )

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
