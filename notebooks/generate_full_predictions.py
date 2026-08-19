"""Generate a complete leakage-safe prediction file from cleaned customer data."""

from pathlib import Path

from retention_recommendation import generate_complete_predictions, prepare_clean_data

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"
RETENTION_FILE = ROOT / "data" / "predictions" / "Retention_Recommendations.csv"


def main():
    data = prepare_clean_data()
    print(f"Loading cleaned vw_ChurnData population: {len(data):,} customers")
    print("Generating 5-fold out-of-fold Random Forest predictions...")

    results = generate_complete_predictions()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    results[results["Predicted_Churn"] == 1].to_csv(RETENTION_FILE, index=False)

    print(f"Total clean customers predicted: {len(results):,}")
    print(f"Predicted churners: {int(results['Predicted_Churn'].sum()):,}")
    print(f"High-risk customers: {int((results['Risk_Level'] == 'High').sum()):,}")
    print(f"Complete prediction output: {OUTPUT_FILE}")
    print(f"Retention output: {RETENTION_FILE}")


if __name__ == "__main__":
    main()
