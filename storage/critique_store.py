"""
storage/critique_store.py — persist CritiqueRecords and run metadata.

Written after every ensemble validation run.
Read by lesson_store.py for distillation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from schemas.validation import CritiqueRecord, ValidationVerdict
from storage.db import get_conn

log = logging.getLogger(__name__)


def write_critique_record(record: CritiqueRecord) -> None:
    """Insert a CritiqueRecord into the database."""
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO critique_records (
                run_uuid, iteration, task_type, task_tags,
                appraisal_critical_count, appraisal_major_count,
                iq2s_inherited_issues, critic_verdicts,
                final_verdict_category, final_verdict_json,
                resolved, total_iterations, loops_triggered,
                ensemble_latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.run_uuid,
                record.iteration,
                record.task_type,
                json.dumps(record.task_tags),
                record.appraisal_critical_count,
                record.appraisal_major_count,
                json.dumps(record.iq2s_inherited_issues),
                json.dumps([v.model_dump() for v in record.critic_verdicts]),
                record.final_verdict.category,
                record.final_verdict.model_dump_json(),
                int(record.resolved),
                record.total_iterations,
                record.loops_triggered,
                record.ensemble_latency_ms,
            ),
        )
        conn.commit()
        log.debug("CritiqueRecord written for run %s iter %d", record.run_uuid, record.iteration)
    finally:
        conn.close()


def write_run(
    run_uuid:   str,
    mode:       str,
    task_type:  str,
    complexity: str,
    is_sub_spec:bool = False,
    parent_run_uuid: Optional[str] = None,
) -> None:
    """Insert a run entry into the runs index table."""
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (
                run_uuid, mode, task_type, complexity,
                is_sub_spec, parent_run_uuid
            ) VALUES (?,?,?,?,?,?)
            """,
            (run_uuid, mode, task_type, complexity,
             int(is_sub_spec), parent_run_uuid),
        )
        conn.commit()
    finally:
        conn.close()


def update_run_status(
    run_uuid:               str,
    status:                 str,
    stage_reached:          Optional[str]   = None,
    correction_iterations:  int             = 0,
    total_tokens:           int             = 0,
    total_latency_ms:       Optional[float] = None,
) -> None:
    """Update a run's status and final metrics."""
    conn = get_conn()
    try:
        conn.execute(
            """
            UPDATE runs SET
                status = ?,
                stage_reached = ?,
                correction_iterations = ?,
                total_tokens = ?,
                total_latency_ms = ?,
                completed_at = strftime('%Y-%m-%dT%H:%M:%S', 'now')
            WHERE run_uuid = ?
            """,
            (status, stage_reached, correction_iterations,
             total_tokens, total_latency_ms, run_uuid),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_critique_records(
    task_type:  Optional[str] = None,
    limit:      int           = 100,
) -> list[dict]:
    """
    Fetch recent critique records for lesson distillation.
    Optionally filter by task_type.
    """
    conn = get_conn()
    try:
        if task_type:
            rows = conn.execute(
                """
                SELECT * FROM critique_records
                WHERE task_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (task_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM critique_records
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_runs_completed() -> int:
    """Return total number of completed pipeline runs."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM runs WHERE status = 'complete'"
        ).fetchone()
        return row["n"] if row else 0
    finally:
        conn.close()
