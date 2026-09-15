"""Ranking, calibration, classification, and financial evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraudshield.threshold import threshold_metrics


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    amounts: np.ndarray | None = None,
    review_cost: float = 5.0,
    loss_rate: float = 1.0,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }
    if amounts is not None:
        result.update(
            threshold_metrics(y, scores, np.asarray(amounts), threshold, review_cost, loss_rate)
        )
    return result


def f1_optimal_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.clip(np.asarray(probabilities, dtype=float), 0, 1))
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])
