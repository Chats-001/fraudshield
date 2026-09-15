# Evaluation results

Generated locally on 2026-09-15 from all 284,807 rows of the public dataset. The final 20% test set
was separated before model search and touched only after model, calibration, and threshold choices.
See `artifacts/model/metrics.json` for the machine-readable source and `artifacts/runs/` for the run.

## Model selection

| Model | CV PR-AUC | Selected parameters |
|---|---:|---|
| Logistic Regression | 0.7594 | C = 0.1 |
| Random Forest | **0.8396** | max depth = 14, min samples leaf = 1 |
| Histogram Gradient Boosting | 0.7567 | learning rate = 0.1, max leaf nodes = 15 |

CV results, not test results, selected the random forest. Sigmoid calibration did not improve the
validation Brier score (0.000598 calibrated vs 0.000575 uncalibrated), so the uncalibrated model was
retained. “Uncalibrated” does not mean probabilities are perfect; ongoing calibration checks remain
necessary.

## Held-out test result

| Model | CV PR-AUC | Test PR-AUC | ROC-AUC | Recall | Precision | Brier | Cost @ 0.1464 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8396 | **0.8265** | 0.9696 | 0.8776 | 0.6232 | 0.000518 | $2,087.50 |

The final confusion matrix was 56,812 true negatives, 52 false positives, 12 false negatives, and
86 true positives. Under the illustrative assumptions, the model captured $8,817.43 of fraud
amount and missed $1,827.50; review cost was $260.00.

## Validation threshold comparison

| Strategy | Threshold | Recall | Precision | Estimated cost |
|---|---:|---:|---:|---:|
| Default | 0.5000 | 0.7342 | 0.8788 | $1,665.79 |
| Maximum F1 | 0.3723 | 0.7722 | 0.8841 | $1,652.72 |
| Cost optimized, recall ≥ 0.80 | **0.1464** | **0.8101** | 0.7191 | $1,735.72 |

The unconstrained lower-cost choices missed the required 80% validation recall. The selected
threshold is therefore the minimum-cost candidate *subject to* the recall constraint, not the
unconstrained cost minimum. Costs assume a $5 legitimate-review cost and 100% loss of missed fraud
amount; they are demonstration inputs rather than industry claims.
