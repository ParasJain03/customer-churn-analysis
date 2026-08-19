"""Validate Use Case 12 dashboard/model metrics against the prediction file.

Run from the repository root:
    python notebooks\\validate_metrics.py

This script checks counts, rates, probabilities, risk thresholds, retention
priority rules, duplicate customer IDs, and actual-vs-predicted churn metrics.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    if not FILE.exists():
        raise FileNotFoundError(f"Missing {FILE}. Run retention_recommendation.py first.")

    df = pd.read_csv(FILE)
    print("=" * 70)
    print("USE CASE 12 METRIC VALIDATION")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # Basic required fields.
    required = [
        "Customer_ID", "Customer_Status", "Churn_Probability",
        "Churn_Probability_Percent", "Predicted_Churn", "Risk_Level",
        "Retention_Priority",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    # 1. Actual historical churn
    # ------------------------------------------------------------------
    actual = df["Customer_Status"].astype(str).str.strip()
    actual_churned = int((actual == "Churned").sum())
    actual_stayed = int((actual == "Stayed").sum())
    actual_known = actual_churned + actual_stayed
    actual_rate = actual_churned / actual_known if actual_known else 0

    print("\n[1] ACTUAL HISTORICAL CHURN")
    print(f"Churned: {actual_churned:,}")
    print(f"Stayed: {actual_stayed:,}")
    print(f"Known status: {actual_known:,}")
    print(f"Actual churn rate = Churned / (Churned + Stayed) = {pct(actual_rate)}")

    # ------------------------------------------------------------------
    # 2. Model predicted churn
    # ------------------------------------------------------------------
    pred = pd.to_numeric(df["Predicted_Churn"], errors="coerce")
    predicted_churners = int((pred == 1).sum())
    predicted_stayed = int((pred == 0).sum())
    predicted_rate = predicted_churners / len(df) if len(df) else 0

    print("\n[2] MODEL PREDICTION")
    print(f"Predicted churners: {predicted_churners:,}")
    print(f"Predicted non-churners: {predicted_stayed:,}")
    print(f"Predicted churn rate = Predicted_Churn==1 / Total = {pct(predicted_rate)}")

    # ------------------------------------------------------------------
    # 3. Probability consistency
    # ------------------------------------------------------------------
    p = pd.to_numeric(df["Churn_Probability"], errors="coerce")
    pp = pd.to_numeric(df["Churn_Probability_Percent"], errors="coerce")
    probability_ok = p.notna().all() and ((p >= 0) & (p <= 1)).all()
    percent_ok = pp.notna().all() and ((pp - p * 100).abs() <= 0.011).all()

    print("\n[3] PROBABILITY FORMULAS")
    print(f"Probability in [0,1]: {'PASS' if probability_ok else 'FAIL'}")
    print(f"Probability_Percent = Probability * 100: {'PASS' if percent_ok else 'FAIL'}")
    print(f"Average churn probability: {p.mean():.4f} ({p.mean() * 100:.2f}%)")

    # ------------------------------------------------------------------
    # 4. Risk thresholds exactly as project code defines them.
    # ------------------------------------------------------------------
    expected_risk = pd.Series("Low", index=df.index)
    expected_risk[p >= 0.40] = "Medium"
    expected_risk[p >= 0.70] = "High"
    risk_ok = (expected_risk.astype(str) == df["Risk_Level"].astype(str)).all()

    print("\n[4] RISK LEVEL FORMULA")
    print("High >= 0.70; Medium >= 0.40 and < 0.70; Low < 0.40")
    print(f"Risk labels: {'PASS' if risk_ok else 'FAIL'}")
    print(df["Risk_Level"].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------------
    # 5. Retention priority consistency.
    # ------------------------------------------------------------------
    expected_priority = pd.Series("Routine monitoring", index=df.index)
    eligible = p >= 0.40
    expected_priority[eligible] = "Medium"
    expected_priority[p >= 0.70] = "High"
    priority_ok = (expected_priority.astype(str) == df["Retention_Priority"].astype(str)).all()

    print("\n[5] RETENTION PRIORITY FORMULA")
    print("High >= 0.70; Medium >= 0.40 and < 0.70; Routine monitoring < 0.40")
    print(f"Priority labels: {'PASS' if priority_ok else 'FAIL'}")
    print(df["Retention_Priority"].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------------
    # 6. Customer ID uniqueness and completeness.
    # ------------------------------------------------------------------
    duplicate_ids = int(df["Customer_ID"].astype(str).duplicated().sum())
    print("\n[6] DATA INTEGRITY")
    print(f"Duplicate Customer_ID rows: {duplicate_ids}")
    print(f"Missing Customer_ID values: {int(df['Customer_ID'].isna().sum())}")
    print(f"Prediction probability missing values: {int(p.isna().sum())}")

    # ------------------------------------------------------------------
    # 7. Gender example used in the AI analyst.
    # ------------------------------------------------------------------
    if "Gender" in df.columns:
        print("\n[7] GENDER EXAMPLE")
        for gender, group in df.groupby("Gender", dropna=False):
            n = len(group)
            churners = int((pd.to_numeric(group["Predicted_Churn"], errors="coerce") == 1).sum())
            rate = churners / n if n else 0
            print(f"{gender}: customers={n:,}, predicted_churners={churners:,}, predicted_churn_rate={pct(rate)}")

    # ------------------------------------------------------------------
    # 8. Confusion-matrix metrics from the current clean population.
    # ------------------------------------------------------------------
    if actual_known == len(df):
        actual_y = (actual == "Churned").astype(int)
        predicted_y = pred.astype(int)
        tp = int(((actual_y == 1) & (predicted_y == 1)).sum())
        tn = int(((actual_y == 0) & (predicted_y == 0)).sum())
        fp = int(((actual_y == 0) & (predicted_y == 1)).sum())
        fn = int(((actual_y == 1) & (predicted_y == 0)).sum())
        accuracy = (tp + tn) / len(df) if len(df) else 0
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        print("\n[8] OUT-OF-FOLD CONFUSION-MATRIX METRICS")
        print(f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
        print(f"Accuracy  = (TP+TN)/N = {accuracy:.4f}")
        print(f"Precision = TP/(TP+FP) = {precision:.4f}")
        print(f"Recall    = TP/(TP+FN) = {recall:.4f}")

    print("\n" + "=" * 70)
    checks = [probability_ok, percent_ok, risk_ok, priority_ok, duplicate_ids == 0, p.isna().sum() == 0]
    print(f"CORE CHECKS: {sum(checks)}/{len(checks)} PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
