"""
storage/chat_store.py — persist chat turns.

A chat is 1:1 with a run: chat_uuid == run_uuid (see api.js). This module
is deliberately independent of the `runs` table — no foreign key, no
cascade — so chats (and the lessons distilled from them) survive a run
row being deleted. See schema_additions.sql for the rationale.
"""

from __future__ import annotations

import logging
from typing import Optional

from storage.db import get_conn

log = logging.getLogger(__name__)


def append_message(
    run_uuid:       str,
    role:           str,             # 'user' | 'assistant' | 'system'
    content:        str,
    node_id:        Optional[str] = None,
    run_iteration:  Optional[int] = None,
) -> int:
    """
    Append a message to a chat, auto-assigning the next seq.
    Returns the assigned seq.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq FROM chats WHERE run_uuid = ?",
            (run_uuid,),
        ).fetchone()
        seq = row["next_seq"]

        conn.execute(
            """
            INSERT INTO chats (run_uuid, seq, role, content, node_id, run_iteration)
            VALUES (?,?,?,?,?,?)
            """,
            (run_uuid, seq, role, content, node_id, run_iteration),
        )
        conn.commit()
        log.debug("Chat message appended: run=%s seq=%d role=%s", run_uuid, seq, role)
        return seq
    finally:
        conn.close()


def get_messages(run_uuid: str) -> list[dict]:
    """Return all messages for a chat, oldest-first, exactly as stored."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM chats WHERE run_uuid = ? ORDER BY seq ASC",
            (run_uuid,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_last_seq(run_uuid: str) -> Optional[int]:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT MAX(seq) AS last_seq FROM chats WHERE run_uuid = ?",
            (run_uuid,),
        ).fetchone()
        return row["last_seq"] if row and row["last_seq"] is not None else None
    finally:
        conn.close()


def format_history_for_prompt(messages: list[dict], max_chars: int = 12_000) -> str:
    """
    Render chat history as plain text for folding into normalised_input
    on a full re-plan turn. Truncates from the FRONT (keeps the most
    recent turns) if the history is long — recent context matters most
    for a follow-up message, and this mirrors MAX_ATTACHMENT_CHARS's
    "guard, don't silently blow the budget" philosophy in server.py.
    """
    if not messages:
        return ""
    lines = []
    for m in messages:
        speaker = "User" if m["role"] == "user" else ("System" if m["role"] == "system" else "Assistant")
        lines.append(f"{speaker}: {m['content']}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        text = "…[earlier turns truncated]…\n\n" + text[-max_chars:]
    return text


def delete_chat(run_uuid: str) -> int:
    """
    Delete all messages for a chat. Does NOT touch lessons or
    lesson_usage — lessons are chat-agnostic by design and must survive
    this. Returns the number of rows deleted.
    """
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM chats WHERE run_uuid = ?", (run_uuid,))
        conn.commit()
        log.debug("Chat deleted: run=%s (%d messages)", run_uuid, cur.rowcount)
        return cur.rowcount
    finally:
        conn.close()