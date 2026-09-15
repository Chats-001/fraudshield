"""FraudShield FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from fraudshield.api.schemas import BatchPrediction, BatchPredictRequest, Prediction, PredictRequest
from fraudshield.api.service import PredictionService
from fraudshield.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = PredictionService.from_artifacts(
            settings.artifact_dir, settings.audit_db
        )
        app.state.startup_error = None
    except (FileNotFoundError, ValueError) as exc:
        app.state.service = None
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="FraudShield API",
    version="0.1.0",
    description="Calibrated fraud probabilities and cost-sensitive review decisions.",
    lifespan=lifespan,
)


def get_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable. Train a model before serving predictions.",
        )
    return service


@app.get("/health")
def health(request: Request) -> dict:
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok" if service else "degraded",
        "model_version": service.model_version if service else None,
    }


@app.post("/predict", response_model=Prediction)
def predict(
    payload: PredictRequest, service: PredictionService = Depends(get_service)
) -> Prediction:
    return service.predict([payload.transaction])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(
    payload: BatchPredictRequest, service: PredictionService = Depends(get_service)
) -> BatchPrediction:
    predictions = service.predict(payload.transactions)
    return BatchPrediction(predictions=predictions, transaction_count=len(predictions))


@app.get("/model/metrics")
def model_metrics(service: PredictionService = Depends(get_service)) -> dict:
    return {
        **service.metrics,
        "model_version": service.model_version,
        "threshold": service.threshold,
    }
