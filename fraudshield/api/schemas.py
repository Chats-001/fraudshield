"""Strict public request and response contracts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudshield.config import settings


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)

    @field_validator("*", mode="after")
    @classmethod
    def finite_numbers_only(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("all feature values must be finite")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: Transaction


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=settings.max_batch_size)


class Prediction(BaseModel):
    fraud_probability: float
    decision_threshold: float
    decision: Literal["approve", "review"]
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    transaction_count: int
