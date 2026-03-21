"""
VisualDescription — produced by the 9B vision decode stage.
Normalises image input into structured text before planning.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class UIElement(BaseModel):
    element_type: str = Field(description="e.g. button, input, table, chart, diagram node")
    label:        Optional[str] = Field(default=None, description="Visible label or text content")
    description:  str  = Field(description="What this element does or represents")


class VisualDescription(BaseModel):
    """
    Structured text description of a visual input.
    Produced by 9B vision decode; consumed by planning stage.
    Downstream models have no awareness of the original input modality.
    """
    summary: str = Field(
        description="One or two sentence summary of what the image shows."
    )
    content_type: str = Field(
        description=(
            "e.g. UI mockup, architecture diagram, ERD, flowchart, "
            "screenshot, wireframe, handwritten sketch"
        )
    )
    ui_elements: list[UIElement] = Field(
        default_factory=list,
        description="Identified UI elements if content_type is UI-related. Empty otherwise."
    )
    extracted_text: list[str] = Field(
        default_factory=list,
        description="Any readable text found in the image, in reading order."
    )
    structural_description: str = Field(
        description=(
            "Detailed description of layout, components, relationships, "
            "and any implied behaviour or data flow."
        )
    )
    inferred_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Requirements or constraints that can be inferred from the visual. "
            "e.g. 'Search must filter results in real time', "
            "'User must be able to add multiple items'."
        )
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description=(
            "Things that are unclear from the image alone and may need "
            "clarification or assumptions during planning."
        )
    )
