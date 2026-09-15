from __future__ import annotations

import sqlite3

import numpy as np
from fastapi.testclient import TestClient

from fraudshield.api.audit import initialize_audit_db
from fraudshield.api.main import app, get_service
from fraudshield.api.service import PredictionService


class FakeModel:
    def predict_proba(self, frame):
        fraud = np.clip(frame["Amount"].to_numpy() / 100.0, 0.0, 1.0)
        return np.column_stack([1 - fraud, fraud])


def make_client(tmp_path):
    database = tmp_path / "audit.sqlite3"
    initialize_audit_db(database)
    service = PredictionService(FakeModel(), "test-v1", 0.5, {"average_precision": 0.9}, database)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), database


def test_health_reports_model(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        app.state.service = app.dependency_overrides[get_service]()
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-v1"}


def test_valid_single_prediction(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/predict", json={"transaction": transaction_dict})
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == 0.75
    assert response.json()["decision"] == "review"


def test_low_score_is_approved(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    transaction_dict["Amount"] = 10
    assert (
        client.post("/predict", json={"transaction": transaction_dict}).json()["decision"]
        == "approve"
    )


def test_batch_prediction(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    other = {**transaction_dict, "Amount": 10}
    response = client.post("/predict/batch", json={"transactions": [transaction_dict, other]})
    assert response.status_code == 200
    assert response.json()["transaction_count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_empty_batch_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_oversized_batch_is_rejected(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    payload = {"transactions": [transaction_dict] * 101}
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_missing_feature_is_rejected(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    del transaction_dict["V17"]
    assert client.post("/predict", json={"transaction": transaction_dict}).status_code == 422


def test_extra_feature_is_rejected(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    transaction_dict["merchant"] = 1
    assert client.post("/predict", json={"transaction": transaction_dict}).status_code == 422


def test_negative_amount_is_rejected(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    transaction_dict["Amount"] = -1
    assert client.post("/predict", json={"transaction": transaction_dict}).status_code == 422


def test_nan_is_rejected(transaction_dict, tmp_path):
    client, _ = make_client(tmp_path)
    transaction_dict["V1"] = "NaN"
    assert client.post("/predict", json={"transaction": transaction_dict}).status_code == 422


def test_metrics_are_pregenerated(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/model/metrics")
    assert response.json()["average_precision"] == 0.9
    assert response.json()["threshold"] == 0.5


def test_prediction_is_audited_without_raw_features(transaction_dict, tmp_path):
    client, database = make_client(tmp_path)
    client.post("/predict", json={"transaction": transaction_dict})
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT model_version, amount FROM prediction_log").fetchone()
        columns = [item[1] for item in connection.execute("PRAGMA table_info(prediction_log)")]
    assert row == ("test-v1", 75.0)
    assert "V1" not in columns
