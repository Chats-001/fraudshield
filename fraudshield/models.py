"""Defensible model candidates and compact parameter grids."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraudshield.config import RANDOM_STATE
from fraudshield.features import make_preprocessor


def model_candidates() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return three intentionally small searches suitable for an imbalanced dataset."""
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__max_depth": [8, 14], "model__min_samples_leaf": [1, 4]},
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            max_iter=200,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
        ),
    }
