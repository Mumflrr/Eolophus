"""
SubSpecInterface — interface contract carried by each sub-spec.
Used by final validation to check cross-spec compatibility.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class InterfaceStatus(str, Enum):
    PENDING   = "pending"
    COMPLETE  = "complete"
    FAILED    = "failed"


class SharedObjectRef(BaseModel):
    name:       str = Field(description="Object name as defined in the top-level PlanSpec")
    defined_in: str = Field(description="Sub-spec UUID or 'root' that defines this object")
    type_hint:  str = Field(description="Type hint as string")
    description:str = Field(description="What this object represents")


class SubSpecInterface(BaseModel):
    """
    Interface contract for a sub-spec within a decomposed task.
    Written alongside each sub-spec's output artefacts.
    Final validation checks all interfaces for compatibility.
    """
    sub_spec_uuid:   str = Field(description="UUID of this sub-spec run")
    parent_run_uuid: str = Field(description="UUID of the top-level decomposed run")
    component_name:  str = Field(description="Which component this sub-spec implements")
    status:          InterfaceStatus = Field(default=InterfaceStatus.PENDING)

    # What this sub-spec consumes
    inputs: list[str] = Field(
        default_factory=list,
        description="Types or named objects this component expects to receive"
    )
    # What this sub-spec produces
    outputs: list[str] = Field(
        default_factory=list,
        description="Types or named objects this component produces"
    )
    # Shared objects referenced from top-level PlanSpec
    shared_object_refs: list[SharedObjectRef] = Field(
        default_factory=list,
        description=(
            "Objects defined in the top-level PlanSpec that this sub-spec references. "
            "Final validation checks these are consistently defined across all sub-specs."
        )
    )
    implementation_path: Optional[str] = Field(
        default=None,
        description="Relative path to the fixed output artefact file"
    )

    model_config = {"use_enum_values": True}
