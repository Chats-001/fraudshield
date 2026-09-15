"""Versioned model bundle persistence with human-readable metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn

from fraudshield.config import FEATURES


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: str | Path, payload: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_model_bundle(
    estimator,
    artifact_dir: str | Path,
    *,
    model_name: str,
    threshold: float,
    metrics: dict,
    dataset_hash: str,
    params: dict | None = None,
) -> dict:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    version = f"{model_name}-{timestamp[:10]}-{dataset_hash[:8]}"
    joblib.dump(estimator, directory / "model.joblib")
    metadata = {
        "model_version": version,
        "model_name": model_name,
        "model_class": estimator.__class__.__name__,
        "training_timestamp": timestamp,
        "features": list(FEATURES),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": dataset_hash,
        "git_commit": git_commit(),
        "params": params or {},
    }
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "threshold.json", {"threshold": threshold})
    write_json(directory / "metrics.json", metrics)
    return metadata


def load_model_bundle(artifact_dir: str | Path) -> tuple[object, dict, float, dict]:
    directory = Path(artifact_dir)
    model = joblib.load(directory / "model.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    threshold = json.loads((directory / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metadata.get("features") != list(FEATURES):
        raise ValueError("Saved model feature schema is incompatible with this service")
    return model, metadata, float(threshold), metrics
