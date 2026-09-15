import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from fraudshield.evaluation import classification_metrics, f1_optimal_threshold


def test_average_precision_matches_sklearn():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.6])
    assert classification_metrics(y, p, 0.5)["average_precision"] == average_precision_score(y, p)


def test_brier_matches_sklearn():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.6])
    assert classification_metrics(y, p, 0.5)["brier_score"] == brier_score_loss(y, p)


def test_confusion_matrix_layout():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.2, 0.9]), 0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_perfect_predictions_have_perfect_ranking():
    metrics = classification_metrics(np.array([0, 1]), np.array([0.0, 1.0]), 0.5)
    assert metrics["average_precision"] == metrics["roc_auc"] == 1.0


def test_f1_threshold_finds_separating_value():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = f1_optimal_threshold(y, p)
    assert threshold == pytest.approx(0.8)
