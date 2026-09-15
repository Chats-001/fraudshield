"""Honest global importance for anonymized PCA-derived features."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def global_permutation_importance(model, x, y, repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(
        model, x, y, scoring="average_precision", n_repeats=repeats, random_state=42, n_jobs=-1
    )
    return pd.DataFrame(
        {
            "feature": x.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
