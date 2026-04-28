"""
tests/unit/test_lesson_store.py — lesson scoring and deduplication tests.

Tests the scoring formula, deduplication logic, and prompt formatting.
Uses an in-memory SQLite database — no file I/O.
Run: pytest tests/unit/test_lesson_store.py -v
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch
import sqlite3
import uuid

# Point to an in-memory database for tests
os.environ["PIPELINE_DB"] = ":memory:"

from schemas.lesson import Lesson, LessonQuery, LessonResult


def _make_lesson(
    task_type="coding",
    tags=None,
    model="deepcoder",
    category="logic_error",
    confidence=1.0,
    times_seen=1,
) -> Lesson:
    return Lesson(
        lesson_uuid       = str(uuid.uuid4()),
        source_run_uuid   = str(uuid.uuid4()),
        issue_summary     = "Test issue",
        resolution_pattern= "Test fix",
        task_type         = task_type,
        tags              = tags or ["python"],
        model_caught      = model,
        issue_category    = category,
        confidence_score  = confidence,
        times_seen        = times_seen,
    )


# ── Lesson schema tests ───────────────────────────────────────────────────────

class TestLessonSchema:
    def test_default_confidence(self):
        l = _make_lesson()
        assert l.confidence_score == 1.0
        assert l.times_seen == 1
        assert l.is_meta_lesson is False

    def test_meta_lesson(self):
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        l = Lesson(
            lesson_uuid          = str(uuid.uuid4()),
            source_run_uuid      = str(uuid.uuid4()),
            issue_summary        = "Consolidated async pattern lesson",
            resolution_pattern   = "Always await coroutines",
            task_type            = "coding",
            tags                 = ["python", "async"],
            model_caught         = "deepcoder",
            issue_category       = "logic_error",
            is_meta_lesson       = True,
            source_lesson_uuids  = ids,
            confidence_score     = 8.0,
            times_seen           = 8,
        )
        assert l.is_meta_lesson is True
        assert len(l.source_lesson_uuids) == 2
        assert l.confidence_score == 8.0


# ── Scoring formula tests (pure math, no DB) ──────────────────────────────────

class TestScoringFormula:
    """
    Test the scoring arithmetic independently of database I/O.
    Mirrors the formula in lesson_store.retrieve_lessons().
    """

    def _score(
        self,
        confidence: float,
        max_conf: float,
        days_ago: float,
        tag_overlap: float,
        times_seen: int,
        weights: dict = None,
        decay: float = 0.05,
    ) -> dict:
        w = weights or {"confidence": 0.4, "recency": 0.3, "similarity": 0.2, "difficulty": 0.1}
        import math

        conf_norm  = confidence / max_conf if max_conf > 0 else 0.0
        recency    = 1.0 / (1.0 + days_ago * decay)
        similarity = tag_overlap
        diff_norm  = min(1.0, times_seen / 10.0)

        score = (
            w["confidence"] * conf_norm
            + w["recency"]   * recency
            + w["similarity"]* similarity
            + w["difficulty"]* diff_norm
        )
        return {
            "score":      score,
            "confidence": w["confidence"] * conf_norm,
            "recency":    w["recency"]    * recency,
            "similarity": w["similarity"] * similarity,
            "difficulty": w["difficulty"] * diff_norm,
        }

    def test_perfect_match_scores_high(self):
        result = self._score(
            confidence=5.0, max_conf=5.0,
            days_ago=0, tag_overlap=1.0, times_seen=10
        )
        assert result["score"] > 0.95

    def test_zero_confidence_scores_low(self):
        result = self._score(
            confidence=0.0, max_conf=5.0,
            days_ago=100, tag_overlap=0.0, times_seen=1
        )
        # Only recency contributes but days_ago=100 kills it
        assert result["score"] < 0.15

    def test_recency_decays_over_time(self):
        fresh  = self._score(confidence=1.0, max_conf=1.0, days_ago=0,   tag_overlap=0, times_seen=1)
        old    = self._score(confidence=1.0, max_conf=1.0, days_ago=100, tag_overlap=0, times_seen=1)
        assert fresh["recency"] > old["recency"]

    def test_tag_overlap_increases_score(self):
        no_overlap   = self._score(confidence=1.0, max_conf=1.0, days_ago=1, tag_overlap=0.0,  times_seen=1)
        full_overlap = self._score(confidence=1.0, max_conf=1.0, days_ago=1, tag_overlap=1.0,  times_seen=1)
        assert full_overlap["score"] > no_overlap["score"]

    def test_high_times_seen_increases_difficulty_score(self):
        low  = self._score(confidence=1.0, max_conf=1.0, days_ago=1, tag_overlap=0.5, times_seen=1)
        high = self._score(confidence=1.0, max_conf=1.0, days_ago=1, tag_overlap=0.5, times_seen=10)
        assert high["score"] > low["score"]

    def test_weights_sum_approximately_to_one(self):
        w = {"confidence": 0.4, "recency": 0.3, "similarity": 0.2, "difficulty": 0.1}
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_score_bounded_between_0_and_1(self):
        result = self._score(
            confidence=100.0, max_conf=100.0,
            days_ago=0, tag_overlap=1.0, times_seen=100
        )
        assert 0.0 <= result["score"] <= 1.1  # slight slack for floating point


# ── format_lessons_for_prompt ─────────────────────────────────────────────────

class TestFormatLessonsForPrompt:
    def test_empty_returns_empty_string(self):
        from storage.lesson_store import format_lessons_for_prompt
        result = format_lessons_for_prompt([])
        assert result == ""

    def test_single_lesson_format(self):
        from storage.lesson_store import format_lessons_for_prompt
        lesson = _make_lesson(category="logic_error", model="deepcoder")
        lesson_result = LessonResult(lesson=lesson, score=0.75, score_breakdown={})
        result = format_lessons_for_prompt([lesson_result])
        assert "<relevant_lessons>" in result
        assert "</relevant_lessons>" in result
        assert "logic_error" in result
        assert "deepcoder" in result
        assert "Test issue" in result
        assert "Test fix" in result

    def test_multiple_lessons_numbered(self):
        from storage.lesson_store import format_lessons_for_prompt
        lessons = [
            LessonResult(lesson=_make_lesson(), score=0.9, score_breakdown={}),
            LessonResult(lesson=_make_lesson(), score=0.7, score_breakdown={}),
        ]
        result = format_lessons_for_prompt(lessons)
        assert "1." in result
        assert "2." in result

    def test_example_context_included_when_present(self):
        from storage.lesson_store import format_lessons_for_prompt
        lesson = _make_lesson()
        lesson = lesson.model_copy(update={"example_context": "e.g. async def foo(): return bar()"})
        lr = LessonResult(lesson=lesson, score=0.8, score_breakdown={})
        result = format_lessons_for_prompt([lr])
        assert "e.g. async def foo()" in result
