"""
storage/lesson_store.py — LessonL experiential memory.

Phase 1: write lessons after each run (store-only, no retrieval).
Phase 2: retrieve top-k relevant lessons for injection into planning prompts.

Scoring function (all weights configurable in routing.yaml):
  score = confidence_weight  * normalised_confidence
        + recency_weight     * recency_decay
        + similarity_weight  * tag_overlap_ratio
        + difficulty_weight  * normalised_difficulty
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from schemas.lesson import Lesson, LessonQuery, LessonResult
from storage.db import get_conn

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_routing_config: dict = {}

def _routing() -> dict:
    if not _routing_config:
        p = Path(__file__).parent.parent / "config" / "routing.yaml"
        with open(p) as f:
            _routing_config.update(yaml.safe_load(f))
    return _routing_config


def _lr() -> dict:
    """Shortcut to lesson_retrieval config block."""
    return _routing()["lesson_retrieval"]


# ── Write ─────────────────────────────────────────────────────────────────────

def write_lesson(lesson: Lesson) -> str:
    """
    Write a lesson to the database.
    Runs deduplication check first — if a sufficiently similar lesson
    exists, increment its confidence score instead of creating a new row.

    Returns the lesson_uuid that was written or updated.
    """
    existing_uuid = _find_duplicate(lesson)
    if existing_uuid:
        _increment_confidence(existing_uuid)
        log.debug("Lesson deduplicated → incremented confidence on %s", existing_uuid)
        return existing_uuid

    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO lessons (
                lesson_uuid, source_run_uuid,
                issue_summary, resolution_pattern, example_context,
                task_type, tags, model_caught, issue_category,
                confidence_score, times_seen, is_meta_lesson,
                source_lesson_uuids, last_triggered
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                lesson.lesson_uuid,
                lesson.source_run_uuid,
                lesson.issue_summary,
                lesson.resolution_pattern,
                lesson.example_context,
                lesson.task_type,
                json.dumps(lesson.tags),
                lesson.model_caught,
                lesson.issue_category,
                lesson.confidence_score,
                lesson.times_seen,
                int(lesson.is_meta_lesson),
                json.dumps(lesson.source_lesson_uuids),
                lesson.last_triggered,
            ),
        )
        conn.commit()
        log.debug("New lesson written: %s", lesson.lesson_uuid)
        return lesson.lesson_uuid
    finally:
        conn.close()


def _find_duplicate(lesson: Lesson) -> Optional[str]:
    """
    Check if a sufficiently similar lesson already exists.
    Similarity: same task_type + same issue_category + same model_caught
                + tag overlap ratio >= dedup_threshold.
    Returns existing lesson_uuid if found, else None.
    """
    threshold = _lr().get("dedup_threshold", 0.85)
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT lesson_uuid, tags FROM lessons
            WHERE task_type = ?
              AND issue_category = ?
              AND model_caught = ?
              AND is_meta_lesson = 0
            """,
            (lesson.task_type, lesson.issue_category, lesson.model_caught),
        ).fetchall()
    finally:
        conn.close()

    new_tags = set(lesson.tags)
    for row in rows:
        existing_tags = set(json.loads(row["tags"]))
        if not existing_tags and not new_tags:
            return row["lesson_uuid"]
        union = existing_tags | new_tags
        if not union:
            continue
        overlap = len(existing_tags & new_tags) / len(union)
        if overlap >= threshold:
            return row["lesson_uuid"]
    return None


def _increment_confidence(lesson_uuid: str) -> None:
    conn = get_conn()
    try:
        conn.execute(
            """
            UPDATE lessons SET
                confidence_score = confidence_score + 1.0,
                times_seen = times_seen + 1,
                last_triggered = strftime('%Y-%m-%dT%H:%M:%S', 'now'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%S', 'now')
            WHERE lesson_uuid = ?
            """,
            (lesson_uuid,),
        )
        conn.commit()
    finally:
        conn.close()


def mark_lesson_useful(lesson_uuid: str) -> None:
    """Call when a retrieved lesson appears in the 9B's planning reasoning."""
    conn = get_conn()
    try:
        conn.execute(
            """
            UPDATE lessons SET
                times_useful = times_useful + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%S', 'now')
            WHERE lesson_uuid = ?
            """,
            (lesson_uuid,),
        )
        conn.commit()
    finally:
        conn.close()


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_lessons(query: LessonQuery) -> list[LessonResult]:
    """
    Retrieve the top-k most relevant lessons for a planning prompt.

    Returns an empty list if:
      - inject_enabled is False in routing.yaml
      - total lesson count is below min_lessons_for_inject
      - no lessons score above min_score
    """
    cfg = _lr()

    if not cfg.get("inject_enabled", False):
        return []

    # Check minimum lesson count
    conn = get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) as n FROM lessons"
        ).fetchone()["n"]
    finally:
        conn.close()

    if total < cfg.get("min_lessons_for_inject", 50):
        log.debug(
            "Lesson injection skipped: %d lessons < threshold %d",
            total, cfg.get("min_lessons_for_inject", 50)
        )
        return []

    # Fetch candidates: task_type must match exactly
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT * FROM lessons
            WHERE task_type = ?
            ORDER BY confidence_score DESC
            LIMIT 200
            """,
            (query.task_type,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    weights   = cfg.get("weights", {})
    w_conf    = weights.get("confidence",  0.4)
    w_rec     = weights.get("recency",     0.3)
    w_sim     = weights.get("similarity",  0.2)
    w_diff    = weights.get("difficulty",  0.1)
    decay     = cfg.get("recency_decay_rate", 0.05)
    min_score = query.min_score

    scored: list[LessonResult] = []
    query_tags = set(query.tags)
    max_conf = max((r["confidence_score"] for r in rows), default=1.0)

    for row in rows:
        lesson_tags = set(json.loads(row["tags"]))

        # Tag overlap ratio
        union = lesson_tags | query_tags
        similarity = len(lesson_tags & query_tags) / len(union) if union else 0.0

        # Recency decay
        last = row["last_triggered"]
        if last:
            try:
                dt     = datetime.fromisoformat(last)
                now    = datetime.now(timezone.utc).replace(tzinfo=None)
                days   = max(0, (now - dt).days)
                recency = 1.0 / (1.0 + days * decay)
            except ValueError:
                recency = 0.5
        else:
            recency = 0.5

        # Normalised confidence
        conf_norm = row["confidence_score"] / max_conf if max_conf > 0 else 0.0

        # Difficulty proxy: high iteration count = harder problem = more valuable lesson
        diff_norm = min(1.0, row["times_seen"] / 10.0)

        score = (
            w_conf * conf_norm
            + w_rec  * recency
            + w_sim  * similarity
            + w_diff * diff_norm
        )

        if score < min_score:
            continue

        lesson = Lesson(
            lesson_uuid          = row["lesson_uuid"],
            source_run_uuid      = row["source_run_uuid"],
            issue_summary        = row["issue_summary"],
            resolution_pattern   = row["resolution_pattern"],
            example_context      = row["example_context"],
            task_type            = row["task_type"],
            tags                 = json.loads(row["tags"]),
            model_caught         = row["model_caught"],
            issue_category       = row["issue_category"],
            confidence_score     = row["confidence_score"],
            times_seen           = row["times_seen"],
            times_retrieved      = row["times_retrieved"],
            times_useful         = row["times_useful"],
            last_triggered       = row["last_triggered"],
            is_meta_lesson       = bool(row["is_meta_lesson"]),
            source_lesson_uuids  = json.loads(row["source_lesson_uuids"]),
        )

        scored.append(LessonResult(
            lesson          = lesson,
            score           = round(score, 4),
            score_breakdown = {
                "confidence": round(w_conf * conf_norm, 4),
                "recency":    round(w_rec  * recency,   4),
                "similarity": round(w_sim  * similarity,4),
                "difficulty": round(w_diff * diff_norm, 4),
            },
        ))

    # Sort descending by score, take top_k
    scored.sort(key=lambda x: x.score, reverse=True)
    top = scored[:query.top_k]

    # Update retrieval counters
    if top:
        conn = get_conn()
        try:
            for r in top:
                conn.execute(
                    """
                    UPDATE lessons SET
                        times_retrieved = times_retrieved + 1,
                        last_triggered  = strftime('%Y-%m-%dT%H:%M:%S','now'),
                        updated_at      = strftime('%Y-%m-%dT%H:%M:%S','now')
                    WHERE lesson_uuid = ?
                    """,
                    (r.lesson.lesson_uuid,),
                )
            conn.commit()
        finally:
            conn.close()

    log.debug(
        "Lesson retrieval: %d candidates → %d returned (top score %.3f)",
        len(rows), len(top), top[0].score if top else 0,
    )
    return top


def format_lessons_for_prompt(lessons: list[LessonResult]) -> str:
    """
    Format retrieved lessons as a <relevant_lessons> block for injection
    immediately before the task description in the planning prompt.
    """
    if not lessons:
        return ""

    lines = ["<relevant_lessons>"]
    for i, r in enumerate(lessons, 1):
        l = r.lesson
        lines.append(f"{i}. [{l.issue_category} | caught by {l.model_caught}]")
        lines.append(f"   Issue: {l.issue_summary}")
        lines.append(f"   Resolution: {l.resolution_pattern}")
        if l.example_context:
            lines.append(f"   Example: {l.example_context}")
    lines.append("</relevant_lessons>")
    return "\n".join(lines)


# ── Count ─────────────────────────────────────────────────────────────────────

def count_lessons() -> int:
    conn = get_conn()
    try:
        return conn.execute("SELECT COUNT(*) as n FROM lessons").fetchone()["n"]
    finally:
        conn.close()
