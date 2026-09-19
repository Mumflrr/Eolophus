"""
api/db_helpers.py — read-only DB query helpers backing the /runs and
/lessons list endpoints.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def get_runs_from_db(limit: int = 100) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def get_lessons_from_db(
    task_type: Optional[str] = None,
    issue_category: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 200,
) -> list[dict]:
    from storage.db import get_conn
    conn = get_conn()
    try:
        conditions = ["confidence_score >= ?"]
        params: list[Any] = [min_confidence]
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if issue_category:
            conditions.append("issue_category = ?")
            params.append(issue_category)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM lessons WHERE {where} ORDER BY confidence_score DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result
    finally:
        conn.close()
