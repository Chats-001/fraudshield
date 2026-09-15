# FraudShield model card

## Intended use

Educational fraud-risk modeling, portfolio demonstration, and discussion of imbalanced
classification and cost-sensitive decisions.

## Not intended for

Real banking production decisions, autonomous account action, or claims about a specific bank's
economics. Human review is assumed for scores above the selected threshold.

## Dataset and target

The model uses one public historical credit-card transaction dataset. Inputs are `Time`, `Amount`,
and anonymized PCA components `V1`–`V28`; the target is binary `Class`. No semantic meaning is
assigned to the anonymized components.

## Evaluation methodology

The final 20% test set is stratified and held out before all model, calibration, and threshold
choices. The remaining development set is split again; model grids use stratified CV on training
data, and calibration/threshold choice use validation data. PR-AUC is primary because positive
examples are rare. ROC-AUC, recall, precision, F1, Brier score, log loss, confusion matrix, and
financial outcomes are secondary.

## Threshold and calibration

The selected threshold minimizes assumed review plus missed-fraud cost while meeting a configurable
recall constraint. Defaults are illustrative, not factual industry estimates. Sigmoid calibration
is retained only when its validation Brier score is no worse than the uncalibrated candidate.

## Monitoring

Offline PSI uses project-specific flags: below 0.10 is `stable`, 0.10–0.25 is `moderate_drift`,
and 0.25 or above is `high_drift`. These are alerting assumptions. Distribution shift does not
automatically imply model failure and should trigger investigation with labeled data.

## Limitations and risks

- One historical dataset may not represent current fraud.
- Anonymized features prevent domain-level causal interpretation.
- Customer, merchant, device, and network context is absent.
- Demographic context is absent, so fairness cannot be meaningfully audited.
- Probabilities can become miscalibrated as behavior changes.
- Financial assumptions and review capacity must be set by the actual operator.
- The demo API is not a tested real-time transaction platform.

