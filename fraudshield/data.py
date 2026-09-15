"""Dataset validation and leakage-safe stratified splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraudshield.config import FEATURES, RANDOM_STATE, TARGET


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise a useful error when the expected public dataset schema is not present."""
    missing = sorted(set((*FEATURES, TARGET)) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame[list(FEATURES)].isna().any().any():
        raise ValueError("Feature values must not contain missing values")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 and 1")
    if (frame["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative")


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame.loc[:, [*FEATURES, TARGET]]


def stratified_train_test_split(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a held-out test split; callers must not use it for tuning."""
    validate_dataset(frame)
    return train_test_split(
        frame.loc[:, list(FEATURES)],
        frame[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET],
    )
