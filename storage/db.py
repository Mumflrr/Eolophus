"""
storage/db.py — SQLite connection management.

Single database file at $PIPELINE_DB (default: ~/.pipeline/pipeline.db).
All storage modules import get_conn() from here.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_DB_PATH: Path | None = None


def get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        env = os.environ.get("PIPELINE_DB")
        if env:
            _DB_PATH = Path(env)
        else:
            _DB_PATH = Path.home() / ".pipeline" / "pipeline.db"
    return _DB_PATH


def get_conn() -> sqlite3.Connection:
    """
    Return a SQLite connection with row_factory set to sqlite3.Row
    (allows column access by name).
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialise() -> None:
    """
    Create all tables if they don't exist.
    Safe to call on every startup.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema_sql = f.read()

    conn = get_conn()
    try:
        conn.executescript(schema_sql)
        conn.commit()
        log.info("Database initialised at %s", get_db_path())
    finally:
        conn.close()
