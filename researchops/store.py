"""Run history in SQLite."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from researchops.config import DB_PATH


def connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "id TEXT PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL, "
        "status TEXT NOT NULL, result TEXT, error TEXT)"
    )
    conn.commit()
    return conn


def start(question: str) -> str:
    run_id = str(uuid4())

    with connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, created_at, question, status) VALUES (?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(), question, "running"),
        )

    return run_id


def finish(run_id: str, result: dict):
    with connection() as conn:
        conn.execute(
            "UPDATE runs SET status=?, result=? WHERE id=?",
            ("completed", json.dumps(result), run_id),
        )


def progress(run_id: str, events: list):
    """Save partial progress while a run is still in progress."""
    with connection() as conn:
        conn.execute(
            "UPDATE runs SET result=? WHERE id=? AND status='running'",
            (json.dumps({"events": events}), run_id),
        )


def fail(run_id: str, error: str):
    with connection() as conn:
        conn.execute(
            "UPDATE runs SET status=?, error=? WHERE id=?",
            ("failed", error[:2000], run_id),
        )


def get(run_id: str):
    with connection() as conn:
        row = conn.execute(
            "SELECT id, created_at, question, status, result, error "
            "FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "created_at": row[1],
        "question": row[2],
        "status": row[3],
        "result": json.loads(row[4]) if row[4] else None,
        "error": row[5],
    }


def recent(limit: int = 20):
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at, question, status FROM runs "
            "ORDER BY created_at DESC LIMIT ?",
            (min(max(limit, 1), 100),),
        ).fetchall()

    return [dict(zip(("id", "created_at", "question", "status"), row, strict=True)) for row in rows]