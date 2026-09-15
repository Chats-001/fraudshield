"""Shared configuration and the versioned transaction schema."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FEATURES = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
TARGET = "Class"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Settings:
    """Runtime paths and business assumptions, configurable with environment variables."""

    artifact_dir: Path = Path(os.getenv("FRAUDSHIELD_ARTIFACT_DIR", "artifacts/model"))
    audit_db: Path = Path(os.getenv("FRAUDSHIELD_AUDIT_DB", "artifacts/predictions.sqlite3"))
    review_cost: float = float(os.getenv("FRAUDSHIELD_REVIEW_COST", "5.0"))
    loss_rate: float = float(os.getenv("FRAUDSHIELD_LOSS_RATE", "1.0"))
    minimum_recall: float = float(os.getenv("FRAUDSHIELD_MINIMUM_RECALL", "0.80"))
    max_batch_size: int = int(os.getenv("FRAUDSHIELD_MAX_BATCH_SIZE", "100"))


settings = Settings()
