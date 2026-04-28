"""
PlanSpec — produced by the 9B planning stage.
Central contract between planning and execution. Contains only viable ideas
that survived the consistency check. The 35B MoE receives this as its primary input.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Parameter(BaseModel):
    name:        str           = Field(description="Parameter name")
    type_hint:   str           = Field(description="Python type hint as string e.g. 'str', 'list[int]', 'Optional[dict]'")
    description: str           = Field(description="What this parameter represents")
    required:    bool          = Field(default=True)
    default:     Optional[str] = Field(default=None, description="Default value as string if not required")


class FunctionSpec(BaseModel):
    name:        str            = Field(description="Function or method name")
    description: str            = Field(description="What this function does")
    parameters:  list[Parameter]= Field(default_factory=list)
    returns:     str            = Field(description="Return type hint as string")
    raises:      list[str]      = Field(default_factory=list, description="Exception types this may raise")
    notes:       Optional[str]  = Field(default=None, description="Implementation notes or constraints")


class ComponentSpec(BaseModel):
    name:           str              = Field(description="Component, class, or module name")
    responsibility: str              = Field(description="Single-sentence description of what this component owns")
    functions:      list[FunctionSpec]= Field(default_factory=list, description="Key functions or methods")
    dependencies:   list[str]        = Field(default_factory=list, description="Other components this depends on")
    data_structures:list[str]        = Field(default_factory=list, description="Key data structures used or owned")
    interface_inputs: list[str]      = Field(default_factory=list, description="What this component consumes from the outside")
    interface_outputs:list[str]      = Field(default_factory=list, description="What this component produces for others")


class DroppedIdea(BaseModel):
    idea:   str = Field(description="Brief description of the dropped idea")
    reason: str = Field(description="Why it was dropped: contradiction / infeasible / far-fetched / out-of-scope")


class PlanSpec(BaseModel):
    """
    Ordered implementation plan produced by the 9B after consistency filtering.
    This is the contract the 35B MoE executes against.
    All downstream models receive this alongside their primary input.
    """
    task_summary: str = Field(
        description="One paragraph summary of what is being built and why."
    )
    chosen_approach: str = Field(
        description="Which approach from ideation (or direct planning) was selected and why."
    )
    components: list[ComponentSpec] = Field(
        description=(
            "Ordered list of components to implement. "
            "Order reflects implementation dependency — earlier items are depended on by later ones."
        )
    )
    implementation_order: list[str] = Field(
        description="Component names in the order they should be implemented."
    )
    shared_data_structures: list[str] = Field(
        default_factory=list,
        description="Data structures used across multiple components — define these first."
    )
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases the implementation must handle explicitly."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Assumptions made during planning. If any are wrong, "
            "the plan may need revision."
        )
    )
    external_dependencies: list[str] = Field(
        default_factory=list,
        description="External libraries or services required."
    )
    moe_routing_context: str = Field(
        description=(
            "Domain context for 35B MoE expert routing. "
            "Include: technology domains, patterns, languages, frameworks. "
            "e.g. 'Python async web API using FastAPI, Pydantic v2, SQLAlchemy 2.0, REST design'"
        )
    )
    dropped_ideas: list[DroppedIdea] = Field(
        default_factory=list,
        description="Ideas from ideation that were filtered out and why. Preserved for lineage tracking."
    )
    confidence_in_plan: str = Field(
        description="high / medium / low — with brief explanation if medium or low."
    )
