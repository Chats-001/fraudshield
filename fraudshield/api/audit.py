"""Minimal SQLite audit log that avoids storing the raw feature vector."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


def initialize_audit_db(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_log (
                prediction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN ('approve', 'review')),
                threshold REAL NOT NULL,
                amount REAL NOT NULL CHECK (amount >= 0)
            )
            """
        )


def log_prediction(
    path: str | Path,
    *,
    model_version: str,
    probability: float,
    decision: str,
    threshold: float,
    amount: float,
) -> str:
    prediction_id = str(uuid.uuid4())
    with sqlite3.connect(path) as connection:
        connection.execute(
            """INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                datetime.now(UTC).isoformat(),
                model_version,
                probability,
                decision,
                threshold,
                amount,
            ),
        )
    return prediction_id
