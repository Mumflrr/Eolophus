"""
pipeline/attachments.py — shared text-composition helper for folding
attachment content into a task string.

Extracted from server.py (which originally defined this inline) so that
pipeline/graph.py's sub_spec_runner_node can reuse the exact same
fencing/truncation logic when building each sub-spec's task_input.
server.py imports pipeline.graph, so pipeline.graph importing back from
server.py would be circular; this module has no FastAPI or server.py
dependency, so both sides can import it cleanly.

server.py still owns _write_attachments and _load_live_attachments (disk
I/O + the AttachmentIn pydantic model) — only the pure text-composition
piece lives here.
"""

from __future__ import annotations

MAX_ATTACHMENT_CHARS = 200_000   # soft sanity cap per file — not a hard product limit,
                                  # just a guard against a pasted-in giant file silently
                                  # blowing the context budget with no feedback to the user
                                  # Kept in sync with server.py's own constant of the same
                                  # name/value — server.py re-exports this one (see below)
                                  # rather than defining a second copy, so there is only
                                  # ever one cap to keep in sync.


def compose_input_with_attachments(task: str, attachments: list[dict]) -> str:
    """
    Fold attached file contents into the text the pipeline actually reads,
    fenced and labeled by filename so nodes can tell task instructions
    apart from file content without needing their own attachment-parsing
    logic.

    Takes plain dicts (not AttachmentIn) with at least {filename, content}
    so it can be called from server.py (request body / _load_live_attachments)
    and from sub_spec_runner_node (state["attachments"]) with the same
    function and identical output shape.
    """
    if not attachments:
        return task
    parts = [task, "", "── Attached files ──"]
    for a in attachments:
        content = a["content"]
        if len(content) > MAX_ATTACHMENT_CHARS:
            content = content[:MAX_ATTACHMENT_CHARS] + "\n… [truncated, file too large]"
        parts.append(f"\n### {a['filename']}\n```\n{content}\n```")
    return "\n".join(parts)