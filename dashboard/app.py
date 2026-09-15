"""Read-only Streamlit companion for generated FraudShield artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fraudshield.api.schemas import Transaction
from fraudshield.config import FEATURES
from fraudshield.monitoring import build_drift_report
from fraudshield.persistence import load_model_bundle

ARTIFACTS = Path("artifacts/model")

st.set_page_config(page_title="FraudShield", layout="wide")
st.title("FraudShield")
st.caption("Cost-sensitive fraud risk exploration; educational use only.")

performance_tab, threshold_tab, score_tab, drift_tab = st.tabs(
    ["Model performance", "Threshold explorer", "Score transaction", "Drift"]
)

if not (ARTIFACTS / "metrics.json").exists():
    st.error("No trained artifact found. Run `python -m scripts.train` first.")
    st.stop()

model, metadata, threshold, metrics = load_model_bundle(ARTIFACTS)

with performance_tab:
    st.subheader(metadata["model_name"].replace("_", " ").title())
    cols = st.columns(4)
    cols[0].metric("Test PR-AUC", f"{metrics['average_precision']:.4f}")
    cols[1].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("Precision", f"{metrics['precision']:.3f}")
    st.write("Confusion matrix", metrics["confusion_matrix"])
    importance_path = ARTIFACTS / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).set_index("feature")
        st.bar_chart(importance["importance_mean"])
        st.caption(
            "Permutation importance; V features are anonymized and have no assigned meaning."
        )

with threshold_tab:
    curve = pd.read_csv(ARTIFACTS / "threshold_curve.csv")
    chosen = st.slider(
        "Decision threshold", float(curve.threshold.min()), float(curve.threshold.max()), threshold
    )
    row = curve.iloc[(curve.threshold - chosen).abs().argsort()[:1]].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Recall", f"{row.recall:.3f}")
    cols[1].metric("Precision", f"{row.precision:.3f}")
    cols[2].metric("Legitimate reviews", f"{int(row.false_positives):,}")
    cols[3].metric("Estimated cost", f"${row.decision_cost:,.0f}")
    st.line_chart(curve.set_index("threshold")[["recall", "precision"]])
    st.caption(f"Trained threshold: {threshold:.4f}. Costs use documented assumptions.")

with score_tab:
    st.write("Enter one transaction. Defaults are neutral demonstration values.")
    values = {}
    first, second, third = st.columns(3)
    for index, feature in enumerate(FEATURES):
        column = (first, second, third)[index % 3]
        minimum = 0.0 if feature == "Amount" else None
        values[feature] = column.number_input(feature, value=0.0, min_value=minimum)
    if st.button("Score transaction"):
        transaction = Transaction(**values)
        probability = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
        decision = "review" if probability >= threshold else "approve"
        st.metric("Fraud probability", f"{probability:.4%}")
        st.write(f"Decision: **{decision}** at threshold {threshold:.4f}")

with drift_tab:
    upload = st.file_uploader("Upload current CSV", type="csv")
    reference_path = ARTIFACTS / "reference.csv"
    if upload and reference_path.exists():
        report = build_drift_report(pd.read_csv(reference_path), pd.read_csv(upload))
        st.write(f"Overall status: **{report['overall_status']}**")
        st.dataframe(pd.DataFrame(report["features"]).T)
        st.info(report["interpretation"])
    else:
        st.caption("Upload a current batch after training to compare monitored distributions.")
