"""Financial cost evaluation and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float = 5.0,
    loss_rate: float = 1.0,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    values = np.asarray(amounts, dtype=float)
    if not (len(y) == len(scores) == len(values)):
        raise ValueError("Labels, probabilities, and amounts must have equal lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if review_cost < 0 or loss_rate < 0:
        raise ValueError("Cost assumptions must be non-negative")
    predicted = scores >= threshold
    true_fraud = y == 1
    tp = int(np.sum(predicted & true_fraud))
    fp = int(np.sum(predicted & ~true_fraud))
    fn = int(np.sum(~predicted & true_fraud))
    tn = int(np.sum(~predicted & ~true_fraud))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    missed_dollars = float(values[~predicted & true_fraud].sum())
    captured_dollars = float(values[predicted & true_fraud].sum())
    cost = fp * review_cost + missed_dollars * loss_rate
    return {
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(precision),
        "recall": float(recall),
        "fraud_dollars_captured": captured_dollars,
        "fraud_dollars_missed": missed_dollars,
        "review_cost": float(fp * review_cost),
        "decision_cost": float(cost),
    }


def threshold_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    thresholds: np.ndarray | None = None,
    review_cost: float = 5.0,
    loss_rate: float = 1.0,
) -> pd.DataFrame:
    candidates = thresholds if thresholds is not None else np.linspace(0.001, 0.5, 200)
    return pd.DataFrame(
        [
            threshold_metrics(y_true, probabilities, amounts, t, review_cost, loss_rate)
            for t in candidates
        ]
    )


def select_cost_threshold(curve: pd.DataFrame, minimum_recall: float | None = 0.8) -> dict:
    eligible = curve if minimum_recall is None else curve[curve["recall"] >= minimum_recall]
    if eligible.empty:
        raise ValueError("No candidate threshold satisfies the minimum recall constraint")
    return (
        eligible.sort_values(["decision_cost", "threshold"], ascending=[True, False])
        .iloc[0]
        .to_dict()
    )
