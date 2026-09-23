"""Scripted stand-ins for the OpenAI client and the objects nodes pass around."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace


class Msg:
    """An assistant message as the OpenAI SDK returns it (content / tool_calls / reasoning_content)."""

    def __init__(self, content=None, tool_calls=None, reasoning=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning

    def model_dump(self, exclude_none=False):
        d = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name, "arguments": t.function.arguments}}
                for t in (self.tool_calls or [])
            ],
        }
        return {k: v for k, v in d.items() if v not in (None, [])} if exclude_none else d


def tool_call(call_id: str, query: str = "", name: str = "search_web", raw_arguments: str | None = None):
    args = raw_arguments if raw_arguments is not None else json.dumps({"query": query})
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def completion(msg: Msg, finish: str = "stop", prompt_tokens: int = 100, completion_tokens: int = 10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def make_openai(script: list):
    """
    Build a fake `OpenAI` class whose chat.completions.create() pops responses
    from `script` (an Exception in the script is raised instead of returned).
    Every call's kwargs are deep-copied into `.calls` — the SDK mutates nothing,
    but the node under test appends to the same `messages` list between calls,
    so a snapshot is the only way to see what each round actually sent.
    """
    class FakeOpenAI:
        calls: list = []
        init_kwargs: list = []

        def __init__(self, **kw):
            FakeOpenAI.init_kwargs.append(kw)
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kw):
            FakeOpenAI.calls.append(copy.deepcopy(kw))
            if not script:
                raise AssertionError("model called more times than the test scripted")
            nxt = script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

    FakeOpenAI.calls = []
    FakeOpenAI.init_kwargs = []
    return FakeOpenAI


class FakePydantic:
    """Minimal stand-in for a pydantic output object (PlanSpec / DraftOutput / IdeationOutput)."""

    def __init__(self, **attrs):
        self.confidence = "high"
        self.clarification_question = None
        self.implementation_order = []
        self.dropped_ideas = []
        self.moe_routing_context = None
        self.component_drafts = []
        self.approaches = []
        self.architectural_directions = []
        self.potential_components = []
        self.__dict__.update(attrs)

    def model_dump_json(self, **kw):
        return "{}"

    def model_copy(self, update=None):
        clone = FakePydantic(**self.__dict__)
        clone.__dict__.update(update or {})
        return clone