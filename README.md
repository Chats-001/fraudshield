# FraudShield

Cost-sensitive credit-card fraud detection and risk scoring. FraudShield modernizes the
CloudAcademy fraud-detection example into a reproducible decision system: it compares models
with PR-AUC, evaluates calibration, selects a review threshold from explicit financial
assumptions, serves versioned scores with FastAPI, audits predictions, and reports drift.

> Educational portfolio project—not intended for real banking decisions.

## Why fraud detection is difficult

Fraud is rare, so accuracy can look excellent while a model misses most fraud. FraudShield
therefore ranks models by cross-validated average precision (PR-AUC), reserves a stratified test
set, and reports recall, precision, ROC-AUC, Brier score, and decision cost. Class weights address
imbalance without leaking synthetic samples across folds.

## Architecture

```text
CSV Dataset -> Validation / Stratified Split -> Three-model CV comparison
                                               |
                                               v
                                  Calibration + Threshold Selection
                                               |
                                               v
                                    Versioned Model Artifact
                                      /         |          \
                               FastAPI      Streamlit    Drift Report
                                  |
                          SQLite Prediction Log -> SQL analytics
```

## Dataset

The expected dataset is the public anonymized European cardholder transaction dataset used by
the original project: `Time`, PCA-derived `V1`–`V28`, `Amount`, and binary `Class`. It is not
committed. Download it using the link in [data/README.md](data/README.md) and place it at
`data/raw/creditcard.csv`.

## Modeling and evaluation

Training compares balanced logistic regression, balanced random forest, and balanced histogram
gradient boosting using three-fold stratified CV and compact grids. Only `Time` and `Amount` are
scaled; the anonymized PCA components are preserved. The top model is evaluated with and without
sigmoid calibration on a validation split. Thresholds are also selected on validation data. The
held-out test set is used once for the final report.

Run the benchmark:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,dashboard]'
python -m scripts.train --data data/raw/creditcard.csv
```

Generated results are recorded in [docs/RESULTS.md](docs/RESULTS.md); machine-readable metrics,
threshold curves, feature importance, model metadata, and run records live under `artifacts/`.

## Cost-sensitive thresholding

The configurable validation objective is:

```text
decision cost = false positives × review cost
              + missed fraud amount × loss rate
```

Defaults (`$5` review, `100%` missed-amount loss, at least `80%` fraud recall) are illustrative
assumptions, not industry facts. Change them with `--review-cost`, `--loss-rate`, and
`--minimum-recall`. The report compares the selected threshold with 0.5 and the F1 optimum.

## API

```bash
uvicorn fraudshield.api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health` — service and model version
- `POST /predict` — one validated 30-feature transaction
- `POST /predict/batch` — 1–100 transactions
- `GET /model/metrics` — frozen evaluation summary

Open `/docs` for the generated request schema. Inputs reject missing/extra fields, negative
amounts, NaN, and infinity. The SQLite audit stores only version, score, decision, threshold, and
amount—not the raw feature vector. Queries are in `sql/`.

## Monitoring and dashboard

```bash
python -m scripts.simulate_current_data
python -m fraudshield.monitoring \
  --reference artifacts/model/reference.csv \
  --current data/processed/current.csv
streamlit run dashboard/app.py
```

The offline report uses PSI bands documented in [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Drift is
a diagnostic signal, not proof that performance degraded. The dashboard presents performance,
threshold economics, transaction scoring, and drift without retraining.

## Tests and containers

```bash
ruff check .
pytest
docker build -t fraudshield .
docker run --rm -p 8000:8000 fraudshield
```

The suite contains 46 focused tests across validation, splitting, cost arithmetic, metrics, API
contracts, audit logging, and monitoring. CI runs lint and tests without retraining.

## Limitations

This single historical, highly anonymized dataset lacks customer, merchant, demographic, and
operational review context. Fairness cannot be meaningfully audited. Costs are hypothetical, and
the service has not been load-tested as real-time banking infrastructure. See the full
[model card](docs/MODEL_CARD.md).

## Attribution

This is a substantial extension and modernization of CloudAcademy's Apache-2.0-licensed
[`fraud-detection`](https://github.com/cloudacademy/fraud-detection) project. Its original logistic
regression training pipeline and Flask endpoint were replaced; the Apache license is preserved.
See [NOTICE](NOTICE) for details.
