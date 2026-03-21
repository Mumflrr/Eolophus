-- schema.sql — SQLite tables for critique records and lessons
-- Run once at startup via storage/db.py initialise()

-- ── Critique records ──────────────────────────────────────────────────────────
-- One row per ensemble run (per correction loop iteration).
-- Stores structured data for longitudinal analysis and lesson distillation.
CREATE TABLE IF NOT EXISTS critique_records (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid                TEXT    NOT NULL,
    iteration               INTEGER NOT NULL DEFAULT 0,
    task_type               TEXT    NOT NULL,
    task_tags               TEXT    NOT NULL DEFAULT '[]',  -- JSON array

    -- Appraisal lineage
    appraisal_critical_count INTEGER NOT NULL DEFAULT 0,
    appraisal_major_count    INTEGER NOT NULL DEFAULT 0,
    iq2s_inherited_issues    TEXT    NOT NULL DEFAULT '[]', -- JSON array of strings

    -- Critic verdicts (serialised full CritiqueVerdict objects)
    critic_verdicts         TEXT    NOT NULL DEFAULT '[]', -- JSON array

    -- Final validation verdict (serialised ValidationVerdict)
    final_verdict_category  TEXT    NOT NULL,
    final_verdict_json      TEXT    NOT NULL,

    -- Outcome
    resolved                INTEGER NOT NULL DEFAULT 0, -- boolean
    total_iterations        INTEGER NOT NULL DEFAULT 0,
    loops_triggered         INTEGER NOT NULL DEFAULT 0,
    ensemble_latency_ms     REAL,

    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_cr_run_uuid   ON critique_records (run_uuid);
CREATE INDEX IF NOT EXISTS idx_cr_task_type  ON critique_records (task_type);
CREATE INDEX IF NOT EXISTS idx_cr_verdict    ON critique_records (final_verdict_category);
CREATE INDEX IF NOT EXISTS idx_cr_created    ON critique_records (created_at);


-- ── Lessons ───────────────────────────────────────────────────────────────────
-- One row per distilled lesson from the LessonL system.
-- Retrieved and injected into planning prompts for similar future tasks.
CREATE TABLE IF NOT EXISTS lessons (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_uuid         TEXT    NOT NULL UNIQUE,
    source_run_uuid     TEXT    NOT NULL,

    -- Content
    issue_summary       TEXT    NOT NULL,
    resolution_pattern  TEXT    NOT NULL,
    example_context     TEXT,

    -- Tagging
    task_type           TEXT    NOT NULL,
    tags                TEXT    NOT NULL DEFAULT '[]', -- JSON array
    model_caught        TEXT    NOT NULL,
    issue_category      TEXT    NOT NULL,

    -- Scoring metadata
    confidence_score    REAL    NOT NULL DEFAULT 1.0,
    times_seen          INTEGER NOT NULL DEFAULT 1,
    times_retrieved     INTEGER NOT NULL DEFAULT 0,
    times_useful        INTEGER NOT NULL DEFAULT 0,
    last_triggered      TEXT,

    -- Meta-distillation
    is_meta_lesson      INTEGER NOT NULL DEFAULT 0, -- boolean
    source_lesson_uuids TEXT    NOT NULL DEFAULT '[]', -- JSON array

    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_lessons_task_type  ON lessons (task_type);
CREATE INDEX IF NOT EXISTS idx_lessons_confidence ON lessons (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_lessons_triggered  ON lessons (last_triggered DESC);
CREATE INDEX IF NOT EXISTS idx_lessons_category   ON lessons (issue_category);
CREATE INDEX IF NOT EXISTS idx_lessons_meta       ON lessons (is_meta_lesson);


-- ── Run index ─────────────────────────────────────────────────────────────────
-- Lightweight index of all pipeline runs for dashboard/analysis.
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid            TEXT    NOT NULL UNIQUE,
    mode                TEXT    NOT NULL,
    task_type           TEXT    NOT NULL,
    complexity          TEXT    NOT NULL,
    is_sub_spec         INTEGER NOT NULL DEFAULT 0,
    parent_run_uuid     TEXT,
    status              TEXT    NOT NULL DEFAULT 'pending',
    stage_reached       TEXT,
    correction_iterations INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL DEFAULT 0,
    total_latency_ms    REAL,
    started_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    completed_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_uuid    ON runs (run_uuid);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_mode    ON runs (mode);
