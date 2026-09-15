"""Lightweight offline drift report using Population Stability Index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def population_stability_index(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI requires non-empty reference and current samples")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("PSI inputs must contain only finite values")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(ref[0], cur) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    epsilon = 1e-6
    ref_pct = np.clip(ref_counts / ref.size, epsilon, None)
    cur_pct = np.clip(cur_counts / cur.size, epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_level(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_drift"
    return "high_drift"


def build_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str] | None = None
) -> dict:
    selected = columns or [
        name
        for name in ["Amount", "Time", "V1", "V2", "V3", "fraud_probability"]
        if name in reference.columns
    ]
    missing = [name for name in selected if name not in current.columns]
    if missing:
        raise ValueError(f"Current data is missing monitored columns: {', '.join(missing)}")
    features = {}
    for name in selected:
        psi = population_stability_index(reference[name], current[name])
        features[name] = {"psi": psi, "status": drift_level(psi)}
    overall = max((item["psi"] for item in features.values()), default=0.0)
    return {
        "overall_status": drift_level(overall),
        "features": features,
        "interpretation": (
            "Distribution drift is a diagnostic signal and does not by itself prove model failure."
        ),
    }


def _load_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current transactions with reference data")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift_report.json"))
    args = parser.parse_args()
    report = build_drift_report(_load_table(args.reference), _load_table(args.current))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({report['overall_status']})")


if __name__ == "__main__":
    main()
