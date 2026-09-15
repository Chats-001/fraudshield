import numpy as np
import pandas as pd
import pytest

from fraudshield.monitoring import build_drift_report, drift_level, population_stability_index


def test_identical_distribution_has_zero_psi():
    values = np.arange(100)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_shifted_distribution_has_larger_psi():
    reference = np.arange(100)
    assert population_stability_index(reference, reference + 100) > 0.25


@pytest.mark.parametrize(
    "psi,status", [(0.01, "stable"), (0.15, "moderate_drift"), (0.3, "high_drift")]
)
def test_drift_levels(psi, status):
    assert drift_level(psi) == status


def test_report_contains_caveat():
    frame = pd.DataFrame({"Amount": np.arange(20), "Time": np.arange(20)})
    report = build_drift_report(frame, frame)
    assert report["overall_status"] == "stable"
    assert "does not" in report["interpretation"]


def test_report_rejects_missing_current_column():
    reference = pd.DataFrame({"Amount": [1, 2], "Time": [1, 2]})
    with pytest.raises(ValueError, match="Time"):
        build_drift_report(reference, pd.DataFrame({"Amount": [1, 2]}))


def test_empty_psi_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [1])
