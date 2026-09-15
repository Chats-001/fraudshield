"""Framework-independent inference service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudshield.api.audit import initialize_audit_db, log_prediction
from fraudshield.api.schemas import Prediction, Transaction
from fraudshield.config import FEATURES
from fraudshield.persistence import load_model_bundle


@dataclass
class PredictionService:
    model: object
    model_version: str
    threshold: float
    metrics: dict
    audit_db: Path

    @classmethod
    def from_artifacts(cls, artifact_dir: Path, audit_db: Path) -> PredictionService:
        model, metadata, threshold, metrics = load_model_bundle(artifact_dir)
        initialize_audit_db(audit_db)
        return cls(model, metadata["model_version"], threshold, metrics, audit_db)

    def predict(self, transactions: list[Transaction]) -> list[Prediction]:
        frame = pd.DataFrame(
            [transaction.model_dump() for transaction in transactions], columns=FEATURES
        )
        probabilities = np.asarray(self.model.predict_proba(frame))[:, 1]
        predictions = []
        for transaction, probability in zip(transactions, probabilities, strict=True):
            score = float(probability)
            decision = "review" if score >= self.threshold else "approve"
            prediction = Prediction(
                fraud_probability=score,
                decision_threshold=self.threshold,
                decision=decision,
                model_version=self.model_version,
            )
            log_prediction(
                self.audit_db,
                model_version=self.model_version,
                probability=score,
                decision=decision,
                threshold=self.threshold,
                amount=transaction.Amount,
            )
            predictions.append(prediction)
        return predictions
