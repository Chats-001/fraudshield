import pytest

from fraudshield.config import FEATURES
from fraudshield.data import load_dataset, stratified_train_test_split, validate_dataset
from fraudshield.features import make_preprocessor


def test_valid_dataset_is_accepted(sample_frame):
    validate_dataset(sample_frame)


def test_missing_target_is_rejected(sample_frame):
    with pytest.raises(ValueError, match="Class"):
        validate_dataset(sample_frame.drop(columns="Class"))


def test_missing_feature_is_rejected(sample_frame):
    with pytest.raises(ValueError, match="V12"):
        validate_dataset(sample_frame.drop(columns="V12"))


def test_non_binary_target_is_rejected(sample_frame):
    sample_frame.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_dataset(sample_frame)


def test_negative_amount_is_rejected(sample_frame):
    sample_frame.loc[0, "Amount"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_dataset(sample_frame)


def test_missing_value_is_rejected(sample_frame):
    sample_frame.loc[0, "V1"] = None
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(sample_frame)


def test_loader_orders_and_drops_extra_columns(sample_frame, tmp_path):
    sample_frame["unexpected"] = 1
    path = tmp_path / "sample.csv"
    sample_frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded.columns.tolist() == [*FEATURES, "Class"]


def test_split_has_no_label_leakage(sample_frame):
    x_train, x_test, _, _ = stratified_train_test_split(sample_frame)
    assert "Class" not in x_train.columns
    assert "Class" not in x_test.columns


def test_split_preserves_fraud_ratio(sample_frame):
    _, _, y_train, y_test = stratified_train_test_split(sample_frame)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)


def test_preprocessor_scales_only_time_and_amount(sample_frame):
    transformed = make_preprocessor().fit_transform(sample_frame[list(FEATURES)])
    assert transformed["Time"].mean() == pytest.approx(0.0, abs=1e-10)
    assert transformed["Amount"].std(ddof=0) == pytest.approx(1.0)
    assert transformed["V1"].equals(sample_frame["V1"])
