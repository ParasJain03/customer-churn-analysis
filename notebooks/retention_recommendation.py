"""Use Case 12: Churn prediction + retention recommendations on cleaned data.

The project uses the cleaned vw_ChurnData population (Churned + Stayed)
for the complete customer analytics population. The Joined-only view is not
used because it reduced the dashboard population to 411 rows.

Model training still uses the existing Random Forest approach. For the
customer-level dashboard predictions, out-of-fold probabilities are generated
so each customer's score is produced by a model that did not train on that
customer. A final model trained on all cleaned data is also saved for future
new-customer inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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

TARGET = "Customer_Status"
DROP_COLUMNS = ["Customer_ID", "Churn_Category", "Churn_Reason"]


def clean_categories(df):
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None").astype(str)
    return df


def prepare_clean_data():
    """Load the cleaned Churned + Stayed customer population."""
    data = pd.read_excel(DATA_FILE, sheet_name="vw_ChurnData")
    data = clean_categories(data)
    data = data[data[TARGET].isin(["Stayed", "Churned"])].copy()
    return data


def fit_encoder_and_transform(train_df, other_df=None):
    train_df = train_df.copy()
    other_df = train_df.copy() if other_df is None else other_df.copy()
    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        if column in train_df.columns:
            encoder = LabelEncoder()
            train_df[column] = train_df[column].fillna("None").astype(str)
            other_df[column] = other_df[column].fillna("None").astype(str)
            encoder.fit(train_df[column])
            known = set(encoder.classes_)
            fallback = encoder.classes_[0]
            other_df[column] = other_df[column].where(other_df[column].isin(known), fallback)
            train_df[column] = encoder.transform(train_df[column])
            other_df[column] = encoder.transform(other_df[column])
            encoders[column] = encoder

    train_df[TARGET] = train_df[TARGET].map({"Stayed": 0, "Churned": 1})
    if TARGET in other_df.columns:
        other_df[TARGET] = other_df[TARGET].map({"Stayed": 0, "Churned": 1})
    return train_df, other_df, encoders


def encode_for_prediction(df, encoders, features):
    df = clean_categories(df.copy())
    for column, encoder in encoders.items():
        if column in df.columns:
            known = set(encoder.classes_)
            fallback = encoder.classes_[0]
            df[column] = df[column].where(df[column].isin(known), fallback)
            df[column] = encoder.transform(df[column])

    df = df.drop(columns=DROP_COLUMNS + [TARGET], errors="ignore")
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
    if probability < 0.40:
        return "Monitor customer; no immediate retention offer required", "Routine monitoring"

    actions, reasons = [], []
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

    priority = "High" if probability >= 0.70 else "Medium"
    recommendation = " | ".join(actions[:3])
    reason_text = ", ".join(reasons[:3])
    return recommendation + f". Key signals: {reason_text}.", priority


def train_model():
    """Train the final Random Forest on all cleaned Churned + Stayed data."""
    raw = prepare_clean_data()
    encoded, _, encoders = fit_encoder_and_transform(raw)
    encoded = encoded.drop(columns=DROP_COLUMNS, errors="ignore")
    X = encoded.drop(columns=TARGET)
    y = encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    # Save the model retrained on all cleaned data for future inference.
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "encoders": encoders, "features": list(X.columns)}, MODEL_FILE)
    return final_model, encoders, list(X.columns)


def generate_oof_predictions(data):
    """Generate leakage-safe predictions for every cleaned customer."""
    data = data.copy()
    X_raw = data.drop(columns=DROP_COLUMNS + [TARGET], errors="ignore")
    y = data[TARGET].map({"Stayed": 0, "Churned": 1}).astype(int).to_numpy()
    probabilities = np.zeros(len(data), dtype=float)
    predictions = np.zeros(len(data), dtype=int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_raw, y), start=1):
        train_raw = data.iloc[train_idx].copy()
        valid_raw = data.iloc[valid_idx].copy()
        train_encoded, valid_encoded, encoders = fit_encoder_and_transform(train_raw, valid_raw)
        train_encoded = train_encoded.drop(columns=DROP_COLUMNS, errors="ignore")
        valid_encoded = valid_encoded.drop(columns=DROP_COLUMNS, errors="ignore")
        X_train = train_encoded.drop(columns=TARGET)
        y_train = train_encoded[TARGET].astype(int)
        X_valid = valid_encoded.drop(columns=TARGET)

        model = RandomForestClassifier(n_estimators=100, random_state=42 + fold, n_jobs=-1)
        model.fit(X_train, y_train)
        probabilities[valid_idx] = model.predict_proba(X_valid)[:, 1]
        predictions[valid_idx] = (probabilities[valid_idx] >= 0.5).astype(int)

    return predictions, probabilities


def generate_complete_predictions():
    data = prepare_clean_data()
    predictions, probabilities = generate_oof_predictions(data)
    results = data.copy()
    results["Churn_Probability"] = probabilities.round(4)
    results["Churn_Probability_Percent"] = (probabilities * 100).round(2)
    results["Predicted_Churn"] = predictions.astype(int)
    results["Risk_Level"] = [get_risk_level(p) for p in probabilities]

    recs = results.apply(
        lambda row: get_retention_recommendation(row, row["Churn_Probability"]),
        axis=1,
        result_type="expand",
    )
    recs.columns = ["Retention_Recommendation", "Retention_Priority"]
    return pd.concat([results, recs], axis=1)


def main():
    print("Loading cleaned vw_ChurnData population...")
    data = prepare_clean_data()
    print(f"Clean customers available: {len(data):,}")

    print("\nTraining Random Forest and saving final model...")
    train_model()

    print("\nGenerating leakage-safe out-of-fold predictions for every cleaned customer...")
    results = generate_complete_predictions()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    retention = results[results["Predicted_Churn"] == 1].copy()
    retention.to_csv(OUTPUT_FILE, index=False)

    full_output = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"
    results.to_csv(full_output, index=False)

    print(f"\nTotal clean customers: {len(results):,}")
    print(f"Predicted churners: {int(results['Predicted_Churn'].sum()):,}")
    print(f"High-risk customers: {int((results['Risk_Level'] == 'High').sum()):,}")
    print(f"Retention output: {OUTPUT_FILE}")
    print(f"Complete prediction output: {full_output}")


if __name__ == "__main__":
    main()
