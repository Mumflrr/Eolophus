"""
schemas/pipeline_def.py — the custom pipeline definition schema.

A PipelineDefinition is data, not code. It describes a graph of steps —
some reusing existing typed nodes (plan, draft, appraise, critic_a, ...),
some freeform (a new system prompt + model + budget, no Python required),
some decision nodes (constrained-output branch points).

This schema is the contract between:
  - the GUI (which will build/edit these)
  - the validator (pipeline/custom_validator.py)
  - the graph compiler (pipeline/custom_graph.py)
  - the API (api/server.py pipeline CRUD endpoints)

Design constraints (deliberate, not oversights):
  - Sequential-with-loop-backs only. No multiple entry points, no diamond
    merges, no node with more than one incoming "content" edge. A decision
    node's multiple outgoing edges are the only branching primitive.
  - Every loop-back edge MUST declare max_iterations. There is no
    auto-detection of cycles — if you wire a loop, you own capping it.
    The validator refuses to save a pipeline with an uncapped cycle.
  - Existing-node reuse carries the existing node's fixed output state key
    (e.g. "plan" always writes plan_spec). Reusing a node twice means the
    second visit overwrites the first. This is intentional — loop-backs
    are retries, not accumulation. If you need to keep both outputs, use
    two freeform nodes instead.
  - Decision nodes never write domain content into state. They only ever
    produce a decision + reasoning, and reasoning is what feeds the
    automatic feedback passthrough described below.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Node types ──────────────────────────────────────────────────────────────

class StepType(str, Enum):
    EXISTING  = "existing"    # reuse a built-in node (plan, draft, appraise, ...)
    FREEFORM  = "freeform"    # new prompt + model + budget, generic text/struct output
    DECISION  = "decision"    # constrained-output branch point


# Names of built-in nodes that can be reused via StepType.EXISTING.
# Each maps to (python_node_function, fixed_state_output_key).
# Kept as a plain tuple list rather than importing the node modules here —
# schemas/ should not import pipeline/ or nodes/ to avoid circular imports.
# The actual function resolution happens in pipeline/custom_graph.py.
EXISTING_NODE_NAMES = [
    "classify", "vision_decode", "ideation", "plan",
    "draft", "draft_short", "appraise", "bugfix",
    "critic_a", "critic_b", "synthesise", "validate",
    "describe", "distiller",
]


# ── Feedback passthrough ─────────────────────────────────────────────────────

class FeedbackMode(str, Enum):
    AUTO = "auto"    # default — decision node's reasoning flows to the next
                      # node's prompt as {feedback}, automatically
    NONE = "none"     # explicit opt-out — next node gets a clean slate


# ── Step definitions ──────────────────────────────────────────────────────────

class ExistingStep(BaseModel):
    """Reuse a built-in node. Optionally override its model/budget."""
    type:      Literal[StepType.EXISTING] = StepType.EXISTING
    id:        str = Field(description="Unique step id within this pipeline")
    node_name: str = Field(description=f"One of: {EXISTING_NODE_NAMES}")

    # Optional overrides — if omitted, uses whatever models.yaml/routing.yaml
    # already assign to this node's role.
    model_override:  Optional[str] = Field(
        default=None, description="Model id to use instead of the configured role default"
    )
    budget_override: Optional[int] = Field(
        default=None, description="Thinking token budget override, -1 = unlimited"
    )

    @field_validator("node_name")
    @classmethod
    def _valid_node(cls, v: str) -> str:
        if v not in EXISTING_NODE_NAMES:
            raise ValueError(f"'{v}' is not a reusable node. Valid: {EXISTING_NODE_NAMES}")
        return v


class FreeformStep(BaseModel):
    """A new node: system prompt + model + budget, generic output shape."""
    type:   Literal[StepType.FREEFORM] = StepType.FREEFORM
    id:     str = Field(description="Unique step id within this pipeline")

    model:         str = Field(description="Model id — must already be downloaded/configured")
    budget_tokens: int = Field(default=0, description="-1 = unlimited, 0 = no thinking")
    thinking:      bool = Field(default=False)

    system_prompt: str = Field(description="Full system prompt text for this node")
    user_template: str = Field(
        default="{input}",
        description=(
            "Template for the user message. {input} is the serialized content "
            "of input_key. {feedback} is available if feedback_mode=auto and "
            "a decision node upstream produced reasoning."
        ),
    )

    input_key:  str = Field(
        default="normalised_input",
        description="State key to read as {input}. Existing-node output keys "
                     "(plan_spec, draft_output, etc.) are valid here.",
    )
    output_key: str = Field(description="State key this step's output is written to")

    feedback_mode: FeedbackMode = Field(default=FeedbackMode.AUTO)


class DecisionOutcome(BaseModel):
    """One possible result of a decision node, and where it routes to."""
    value:      str = Field(description="The outcome label, e.g. 'pass', 'retry'")
    next_step:  str = Field(description="Step id to route to when this outcome fires, "
                                          "or '__end__' to end the pipeline")
    description: Optional[str] = Field(
        default=None, description="Shown in the GUI edge label; helps the model prompt too"
    )


class DecisionStep(BaseModel):
    """
    A branch point. Reads prior state, asks a model a constrained question,
    routes based on the answer. Never writes domain content — only ever
    produces {decision, reasoning}.
    """
    type: Literal[StepType.DECISION] = StepType.DECISION
    id:   str = Field(description="Unique step id within this pipeline")

    model:         str  = Field(default="9b", description="Defaults to 9b — decisions are cheap")
    thinking:      bool = Field(default=False)
    budget_tokens: int  = Field(default=0)

    system_prompt: str = Field(description="What to evaluate and how to decide")
    input_key:     str = Field(
        default="normalised_input",
        description="State key whose content is presented to the decision model",
    )

    outcomes: list[DecisionOutcome] = Field(
        min_length=2,
        description="All possible outcomes and their routing. Must be exhaustive — "
                     "the model's answer is validated against these values.",
    )

    # Loop-back safety — REQUIRED if any outcome routes to a step that
    # appears earlier in the pipeline (i.e. this decision creates a cycle).
    # The validator computes this automatically and refuses to save if a
    # cycle exists with is_loop_back unset or max_iterations unset.
    is_loop_back:   bool           = Field(default=False)
    max_iterations: Optional[int]  = Field(
        default=None,
        description="Required when is_loop_back=True. Pipeline halts as "
                     "'unresolvable' if exceeded.",
    )

    feedback_mode: FeedbackMode = Field(default=FeedbackMode.AUTO)

    @model_validator(mode="after")
    def _loop_back_needs_cap(self) -> "DecisionStep":
        if self.is_loop_back and not self.max_iterations:
            raise ValueError(
                f"Decision step '{self.id}' is marked is_loop_back=True but has "
                f"no max_iterations. Every loop-back must declare a cap."
            )
        return self


PipelineStep = ExistingStep | FreeformStep | DecisionStep


# ── Pipeline definition ────────────────────────────────────────────────────────

class PipelineDefinition(BaseModel):
    """
    A complete custom pipeline. Stored as one YAML/JSON file under
    config/pipelines/{name}.yaml.
    """
    name:        str = Field(description="Unique pipeline identifier, used in /run's pipeline field")
    description: str = Field(default="", description="Human-readable summary for the GUI")

    entry_step: str = Field(description="Step id to start execution at")
    steps:      list[PipelineStep] = Field(min_length=1)

    # Linear default edges: step N -> step N+1 in list order, UNLESS the
    # step is a DecisionStep (whose outcomes define its own routing) or
    # is explicitly overridden here.
    edge_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="step_id -> next_step_id, overrides the default 'next item "
                     "in list' edge for non-decision steps. '__end__' ends the pipeline.",
    )

    max_total_iterations: int = Field(
        default=20,
        description="Hard safety cap across ALL loop-backs combined, independent "
                     "of any individual decision node's max_iterations. Prevents "
                     "runaway pipelines even if per-loop caps are set generously.",
    )

    @model_validator(mode="after")
    def _ids_unique(self) -> "PipelineDefinition":
        ids = [s.id for s in self.steps]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate step ids: {dupes}")
        return self

    @model_validator(mode="after")
    def _entry_exists(self) -> "PipelineDefinition":
        ids = {s.id for s in self.steps}
        if self.entry_step not in ids:
            raise ValueError(f"entry_step '{self.entry_step}' is not a defined step")
        return self
