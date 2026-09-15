"""Train, compare, calibrate, threshold, evaluate, and persist FraudShield."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from fraudshield.calibration import calibrate_model
from fraudshield.config import RANDOM_STATE, settings
from fraudshield.data import load_dataset, stratified_train_test_split
from fraudshield.evaluation import classification_metrics, f1_optimal_threshold
from fraudshield.explain import global_permutation_importance
from fraudshield.models import model_candidates
from fraudshield.persistence import git_commit, save_model_bundle, sha256_file, write_json
from fraudshield.threshold import select_cost_threshold, threshold_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--artifacts", type=Path, default=settings.artifact_dir)
    parser.add_argument("--review-cost", type=float, default=settings.review_cost)
    parser.add_argument("--loss-rate", type=float, default=settings.loss_rate)
    parser.add_argument("--minimum-recall", type=float, default=settings.minimum_recall)
    return parser.parse_args()


def _search_models(x_train, y_train) -> tuple[dict, list[dict]]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fitted: dict[str, GridSearchCV] = {}
    comparison = []
    for name, (pipeline, grid) in model_candidates().items():
        search = GridSearchCV(
            pipeline, grid, scoring="average_precision", cv=cv, n_jobs=-1, refit=True
        ).fit(x_train, y_train)
        fitted[name] = search
        comparison.append(
            {
                "model": name,
                "cv_average_precision": float(search.best_score_),
                "params": search.best_params_,
            }
        )
        print(f"{name}: CV PR-AUC={search.best_score_:.6f}")
    return fitted, comparison


def train(args: argparse.Namespace) -> dict:
    frame = load_dataset(args.data)
    x_development, x_test, y_development, y_test = stratified_train_test_split(frame)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_development,
        y_development,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )
    searches, comparison = _search_models(x_train, y_train)
    best_row = max(comparison, key=lambda row: row["cv_average_precision"])
    uncalibrated = searches[best_row["model"]].best_estimator_
    raw_validation = uncalibrated.predict_proba(x_validation)[:, 1]
    calibrated = calibrate_model(clone(uncalibrated), x_train, y_train, method="sigmoid", cv=3)
    calibrated_validation = calibrated.predict_proba(x_validation)[:, 1]
    raw_brier = brier_score_loss(y_validation, raw_validation)
    calibrated_brier = brier_score_loss(y_validation, calibrated_validation)
    if calibrated_brier <= raw_brier:
        selected_model, validation_scores, calibration = (
            calibrated,
            calibrated_validation,
            "sigmoid",
        )
    else:
        selected_model, validation_scores, calibration = uncalibrated, raw_validation, "none"

    curve = threshold_curve(
        y_validation.to_numpy(),
        validation_scores,
        x_validation["Amount"].to_numpy(),
        review_cost=args.review_cost,
        loss_rate=args.loss_rate,
    )
    selected = select_cost_threshold(curve, args.minimum_recall)
    f1_threshold = f1_optimal_threshold(y_validation.to_numpy(), validation_scores)
    threshold_comparison = {
        "default_0_5": classification_metrics(
            y_validation.to_numpy(),
            validation_scores,
            0.5,
            x_validation["Amount"].to_numpy(),
            args.review_cost,
            args.loss_rate,
        ),
        "maximum_f1": classification_metrics(
            y_validation.to_numpy(),
            validation_scores,
            f1_threshold,
            x_validation["Amount"].to_numpy(),
            args.review_cost,
            args.loss_rate,
        ),
        "cost_optimized": classification_metrics(
            y_validation.to_numpy(),
            validation_scores,
            selected["threshold"],
            x_validation["Amount"].to_numpy(),
            args.review_cost,
            args.loss_rate,
        ),
    }
    test_scores = selected_model.predict_proba(x_test)[:, 1]
    final_metrics = classification_metrics(
        y_test.to_numpy(),
        test_scores,
        selected["threshold"],
        x_test["Amount"].to_numpy(),
        args.review_cost,
        args.loss_rate,
    )
    final_metrics.update(
        {
            "model_name": best_row["model"],
            "cv_average_precision": best_row["cv_average_precision"],
            "calibration": calibration,
            "uncalibrated_validation_brier": float(raw_brier),
            "calibrated_validation_brier": float(calibrated_brier),
            "cost_assumptions": {
                "review_cost": args.review_cost,
                "loss_rate": args.loss_rate,
                "minimum_recall": args.minimum_recall,
            },
            "threshold_comparison_validation": threshold_comparison,
            "model_comparison": comparison,
        }
    )
    dataset_hash = sha256_file(args.data)
    metadata = save_model_bundle(
        selected_model,
        args.artifacts,
        model_name=best_row["model"],
        threshold=selected["threshold"],
        metrics=final_metrics,
        dataset_hash=dataset_hash,
        params=best_row["params"],
    )
    curve.to_csv(args.artifacts / "threshold_curve.csv", index=False)
    importance_sample = x_validation.sample(
        min(5_000, len(x_validation)), random_state=RANDOM_STATE
    )
    global_permutation_importance(
        selected_model, importance_sample, y_validation.loc[importance_sample.index], repeats=3
    ).to_csv(args.artifacts / "feature_importance.csv", index=False)
    reference = x_train.sample(min(10_000, len(x_train)), random_state=RANDOM_STATE).copy()
    reference["fraud_probability"] = selected_model.predict_proba(reference)[:, 1]
    reference.to_csv(args.artifacts / "reference.csv", index=False)
    run = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "model_name": best_row["model"],
        "params": best_row["params"],
        "cv_average_precision": best_row["cv_average_precision"],
        "test_average_precision": final_metrics["average_precision"],
        "roc_auc": final_metrics["roc_auc"],
        "threshold": selected["threshold"],
        "decision_cost": final_metrics["decision_cost"],
        "model_version": metadata["model_version"],
    }
    write_json(Path("artifacts/runs") / f"{run['run_id']}.json", run)
    print(json.dumps(run, indent=2))
    return run


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
