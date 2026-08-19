"""Use Case 12 extension for the existing telecom churn project.

Keeps the existing Random Forest approach and adds:
1. Churn probability
2. Risk segmentation
3. Retention priority
4. Rule-based personalized retention recommendations
5. CSV output for Power BI / downstream GenAI

Input: data/processed/Prediction_Data.xlsx
Sheets: vw_ChurnData and vw_JoinData
Output: data/predictions/Retention_Recommendations.csv
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "Prediction_Data.xlsx"
OUTPUT_FILE = ROOT / "data" / "predictions" / "Retention_Recommendations.csv"
MODEL_FILE = ROOT / "data" / "predictions" / "churn_model.joblib"

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "State", "Value_Deal", "Phone_Service",
    "Multiple_Lines", "Internet_Service", "Internet_Type", "Online_Security",
    "Online_Backup", "Device_Protection_Plan", "Premium_Support",
    "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    "Contract", "Paperless_Billing", "Payment_Method"
]


def clean_categories(df):
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None").astype(str)
    return df


def train_model():
    data = pd.read_excel(DATA_FILE, sheet_name="vw_ChurnData")
    data = clean_categories(data)

    # Same target/leakage treatment as the existing GitHub notebook.
    data = data.drop(
        columns=["Customer_ID", "Churn_Category", "Churn_Reason"],
        errors="ignore"
    )

    encoders = {}
    for column in CATEGORICAL_COLUMNS:
        if column in data.columns:
            encoders[column] = LabelEncoder()
            data[column] = encoders[column].fit_transform(data[column])

    data["Customer_Status"] = data["Customer_Status"].map(
        {"Stayed": 0, "Churned": 1}
    )

    data = data.dropna(subset=["Customer_Status"])

    X = data.drop(columns="Customer_Status")
    y = data["Customer_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "encoders": encoders, "features": list(X.columns)},
        MODEL_FILE
    )

    return model, encoders, list(X.columns)


def encode_for_prediction(df, encoders, features):
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None").astype(str)
            if column in encoders:
                # Existing project uses LabelEncoder. Unknown values are mapped
                # to the first known class so inference does not crash.
                known = set(encoders[column].classes_)
                fallback = encoders[column].classes_[0]
                df[column] = df[column].where(df[column].isin(known), fallback)
                df[column] = encoders[column].transform(df[column])

    df = df.drop(
        columns=["Customer_ID", "Customer_Status", "Churn_Category", "Churn_Reason"],
        errors="ignore"
    )

    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    return df[features]


def get_risk_level(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def get_retention_recommendation(row, probability):
    """Business-rule layer: prediction -> actionable retention action."""
    if probability < 0.40:
        return (
            "Monitor customer; no immediate retention offer required",
            "Routine monitoring"
        )

    actions = []
    reasons = []

    contract = str(row.get("Contract", ""))
    tenure = pd.to_numeric(row.get("Tenure_in_Months", 0), errors="coerce")
    monthly_charge = pd.to_numeric(row.get("Monthly_Charge", 0), errors="coerce")
    premium_support = str(row.get("Premium_Support", ""))
    internet_type = str(row.get("Internet_Type", ""))
    value_deal = str(row.get("Value_Deal", ""))

    if contract == "Month-to-Month":
        actions.append("Offer an incentive to move to a long-term contract")
        reasons.append("month-to-month contract")

    if pd.notna(monthly_charge) and monthly_charge >= 80:
        actions.append("Offer a personalized pricing or loyalty discount")
        reasons.append("high monthly charge")

    if pd.notna(tenure) and tenure < 12:
        actions.append("Offer an early-tenure loyalty benefit")
        reasons.append("short customer tenure")

    if premium_support.lower() in {"no", "none", "nan"}:
        actions.append("Offer priority/premium customer support")
        reasons.append("premium support not active")

    if internet_type.lower() in {"fiber optic", "fiber"}:
        actions.append("Offer a service review or upgrade consultation")
        reasons.append("fiber service customer")

    if value_deal.lower() in {"none", "nan", ""}:
        actions.append("Evaluate eligibility for a value plan")
        reasons.append("no value deal assigned")

    if not actions:
        actions.append("Assign customer to a proactive retention campaign")
        reasons.append("elevated churn probability")

    if probability >= 0.70:
        priority = "High"
    else:
        priority = "Medium"

    # Keep the output concise for Power BI and future GenAI use.
    recommendation = " | ".join(actions[:3])
    reason_text = ", ".join(reasons[:3])
    return recommendation + f". Key signals: {reason_text}.", priority


def predict_joined_customers(model, encoders, features):
    joined = pd.read_excel(DATA_FILE, sheet_name="vw_JoinData")
    joined_clean = clean_categories(joined)
    X_joined = encode_for_prediction(joined_clean, encoders, features)

    predictions = model.predict(X_joined)
    probabilities = model.predict_proba(X_joined)[:, 1]

    results = joined.copy()
    results["Churn_Probability"] = probabilities.round(4)
    results["Churn_Probability_Percent"] = (probabilities * 100).round(2)
    results["Predicted_Churn"] = predictions
    results["Risk_Level"] = [get_risk_level(p) for p in probabilities]

    recommendation_data = results.apply(
        lambda row: get_retention_recommendation(
            row, row["Churn_Probability"]
        ),
        axis=1,
        result_type="expand"
    )
    recommendation_data.columns = [
        "Retention_Recommendation", "Retention_Priority"
    ]

    results = pd.concat([results, recommendation_data], axis=1)

    # Retention campaigns focus on predicted churners.
    results = results[results["Predicted_Churn"] == 1].copy()
    results = results.sort_values(
        ["Retention_Priority", "Churn_Probability"],
        ascending=[True, False]
    )

    return results


def main():
    print("Training existing Random Forest churn model...")
    model, encoders, features = train_model()

    print("\nGenerating Use Case 12 retention recommendations...")
    results = predict_joined_customers(model, encoders, features)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nPredicted churners: {len(results)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nPreview:")
    print(
        results[
            [
                "Customer_ID",
                "Churn_Probability_Percent",
                "Risk_Level",
                "Retention_Priority",
                "Retention_Recommendation",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
