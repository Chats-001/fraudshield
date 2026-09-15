from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.config import FEATURES, TARGET


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    frame["Time"] = np.arange(200)
    frame["Amount"] = rng.uniform(0, 500, 200)
    frame[TARGET] = np.array([0] * 180 + [1] * 20)
    return frame


@pytest.fixture
def transaction_dict() -> dict[str, float]:
    values = {"Time": 123.0, "Amount": 75.0}
    values.update({f"V{i}": 0.0 for i in range(1, 29)})
    return values
