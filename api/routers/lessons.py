"""
api/routers/lessons.py — lesson store listing/deletion, and the
not-yet-implemented distillation trigger.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.db_helpers import get_lessons_from_db

router = APIRouter()


@router.get("/lessons")
async def get_lessons(
    task_type:      Optional[str]   = None,
    issue_category: Optional[str]   = None,
    min_confidence: float           = 0.0,
    limit:          int             = 100,
):
    """List lessons with optional filters."""
    return get_lessons_from_db(task_type, issue_category, min_confidence, limit)


@router.delete("/lessons/{lesson_uuid}")
async def delete_lesson(lesson_uuid: str):
    """Remove a lesson from the store."""
    from storage.db import get_conn
    conn = get_conn()
    try:
        conn.execute("DELETE FROM lessons WHERE lesson_uuid = ?", (lesson_uuid,))
        conn.commit()
        if conn.execute(
            "SELECT changes() as n"
        ).fetchone()["n"] == 0:
            raise HTTPException(status_code=404, detail="Lesson not found")
    finally:
        conn.close()
    return {"status": "deleted", "lesson_uuid": lesson_uuid}


@router.post("/lessons/distill")
async def trigger_distillation():
    """Manually trigger meta-distillation of accumulated lessons."""
    # Placeholder — meta-distillation node not yet implemented
    return {"status": "not_implemented", "message": "Meta-distillation coming soon"}
