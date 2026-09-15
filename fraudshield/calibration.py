"""Probability calibration utilities that never consume the held-out test set."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def calibrate_model(estimator, x_train, y_train, method: str = "sigmoid", cv: int = 3):
    calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
    return calibrated.fit(x_train, y_train)


def calibration_summary(y_true, probabilities, bins: int = 10) -> dict:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "mean_predicted_probability": predicted.tolist(),
        "observed_fraud_rate": observed.tolist(),
    }
