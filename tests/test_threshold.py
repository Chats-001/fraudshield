import numpy as np
import pandas as pd
import pytest

from fraudshield.threshold import select_cost_threshold, threshold_curve, threshold_metrics

Y = np.array([0, 0, 1, 1])
P = np.array([0.1, 0.6, 0.4, 0.9])
A = np.array([10.0, 20.0, 100.0, 200.0])


def test_threshold_confusion_counts():
    result = threshold_metrics(Y, P, A, 0.5)
    assert (result["true_positives"], result["false_positives"]) == (1, 1)
    assert (result["false_negatives"], result["true_negatives"]) == (1, 1)


def test_threshold_precision_and_recall():
    result = threshold_metrics(Y, P, A, 0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_financial_cost_uses_missed_fraud_amount():
    result = threshold_metrics(Y, P, A, 0.5, review_cost=5, loss_rate=0.5)
    assert result["review_cost"] == 5
    assert result["fraud_dollars_missed"] == 100
    assert result["decision_cost"] == 55


def test_fraud_dollars_captured():
    assert threshold_metrics(Y, P, A, 0.5)["fraud_dollars_captured"] == 200


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        threshold_metrics(Y, P[:-1], A, 0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="between"):
        threshold_metrics(Y, P, A, threshold)


def test_negative_cost_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        threshold_metrics(Y, P, A, 0.5, review_cost=-1)


def test_curve_has_one_row_per_threshold():
    curve = threshold_curve(Y, P, A, thresholds=np.array([0.2, 0.5, 0.8]))
    assert curve["threshold"].tolist() == [0.2, 0.5, 0.8]


def test_selection_minimizes_cost_subject_to_recall():
    curve = pd.DataFrame(
        {"threshold": [0.1, 0.2, 0.3], "recall": [1.0, 0.9, 0.7], "decision_cost": [30, 10, 1]}
    )
    assert select_cost_threshold(curve, minimum_recall=0.8)["threshold"] == 0.2


def test_impossible_recall_constraint_is_rejected():
    curve = pd.DataFrame({"threshold": [0.5], "recall": [0.5], "decision_cost": [1]})
    with pytest.raises(ValueError, match="No candidate"):
        select_cost_threshold(curve, minimum_recall=0.9)
