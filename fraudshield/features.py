"""Preprocessing that scales only the non-PCA transaction fields."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from fraudshield.config import FEATURES


def make_preprocessor() -> ColumnTransformer:
    """Scale Time and Amount while preserving already transformed V features."""
    return ColumnTransformer(
        [("scale", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def transformed_feature_names() -> tuple[str, ...]:
    return ("Time", "Amount", *(name for name in FEATURES if name not in {"Time", "Amount"}))
