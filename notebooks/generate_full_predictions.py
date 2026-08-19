"""Generate predictions for every customer for the AI Data Q&A layer.

This keeps the existing retention CSV focused on predicted churners while also
creating a complete customer-level prediction file for questions about all rows.
"""

from pathlib import Path
import joblib
import pandas as pd

from retention_recommendation import (
    CATEGORICAL_COLUMNS,
    DATA_FILE,
    MODEL_FILE,
    clean_categories,
    encode_for_prediction,
    get_risk_level,
    get_retention_recommendation,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"


def main():
    print("Training existing Random Forest model...")
    model, encoders, features = train_model()

    print("Generating predictions for every customer...")
    joined = pd.read_excel(DATA_FILE, sheet_name="vw_JoinData")
    joined_clean = clean_categories(joined)
    X_joined = encode_for_prediction(joined_clean, encoders, features)

    predictions = model.predict(X_joined)
    probabilities = model.predict_proba(X_joined)[:, 1]

    results = joined.copy()
    results["Churn_Probability"] = probabilities.round(4)
    results["Churn_Probability_Percent"] = (probabilities * 100).round(2)
    results["Predicted_Churn"] = predictions.astype(int)
    results["Risk_Level"] = [get_risk_level(p) for p in probabilities]

    recommendations = results.apply(
        lambda row: get_retention_recommendation(row, row["Churn_Probability"]),
        axis=1,
        result_type="expand",
    )
    recommendations.columns = ["Retention_Recommendation", "Retention_Priority"]
    results = pd.concat([results, recommendations], axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Total customers predicted: {len(results):,}")
    print(f"Predicted churners: {int(results['Predicted_Churn'].sum()):,}")
    print(f"High-risk customers: {int((results['Risk_Level'] == 'High').sum()):,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
