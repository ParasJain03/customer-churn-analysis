"""Streamlit demo for Use Case 12: Telecom Churn + Retention Recommendation."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "predictions" / "Retention_Recommendations.csv"

load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Telecom Retention AI", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Retention AI")
st.caption("Use Case 12 — Churn Prediction + Personalized Retention Recommendation")


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE)


df = load_data()

if df.empty:
    st.error("Retention_Recommendations.csv was not found.")
    st.code("python notebooks\\retention_recommendation.py")
    st.stop()

required_columns = {
    "Customer_ID", "Churn_Probability", "Churn_Probability_Percent",
    "Predicted_Churn", "Risk_Level", "Retention_Priority",
    "Retention_Recommendation",
}
missing = required_columns.difference(df.columns)
if missing:
    st.error(f"Missing columns: {', '.join(sorted(missing))}")
    st.stop()

st.sidebar.header("Customer Filters")
filtered = df.copy()
for column in ["Risk_Level", "Retention_Priority", "Contract", "State", "Internet_Type"]:
    if column in filtered.columns:
        values = sorted(filtered[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(column.replace("_", " "), values, default=values)
        filtered = filtered[filtered[column].astype(str).isin(selected)]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Predicted Churners", f"{len(filtered):,}")
with c2:
    st.metric("High-Risk Customers", f"{(filtered['Risk_Level'] == 'High').sum():,}")
with c3:
    st.metric("High-Priority Customers", f"{(filtered['Retention_Priority'] == 'High').sum():,}")
with c4:
    avg_probability = filtered["Churn_Probability"].mean() if not filtered.empty else 0
    st.metric("Average Churn Probability", f"{avg_probability:.1%}")

st.divider()

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

customer_ids = filtered["Customer_ID"].astype(str).tolist()
selected_id = st.selectbox("Select a customer", customer_ids)
customer = filtered[filtered["Customer_ID"].astype(str) == selected_id].iloc[0]

left, right = st.columns([1, 1])
with left:
    st.subheader("Customer Risk")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Churn Probability", f"{customer['Churn_Probability_Percent']:.1f}%")
    with r2:
        st.metric("Risk Level", str(customer["Risk_Level"]))
    st.write(f"**Retention Priority:** {customer['Retention_Priority']}")
    st.write(f"**Predicted Churn:** {'Yes' if int(customer['Predicted_Churn']) == 1 else 'No'}")

with right:
    st.subheader("Customer Profile")
    profile_columns = [
        "State", "Contract", "Tenure_in_Months", "Monthly_Charge",
        "Internet_Type", "Premium_Support", "Value_Deal", "Payment_Method"
    ]
    profile = {
        c.replace("_", " "): str(customer[c])
        for c in profile_columns if c in customer.index
    }
    profile_df = pd.DataFrame(list(profile.items()), columns=["Attribute", "Value"])
    st.dataframe(profile_df, width="stretch", hide_index=True)

st.subheader("🎯 Retention Recommendation")
st.info(str(customer["Retention_Recommendation"]))

st.subheader("🤖 AI Retention Strategy")
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

if gemini_key and genai is not None:
    if st.button("Generate Personalized AI Strategy", type="primary"):
        system_prompt = (
            "You are a telecom customer-retention assistant. Generate a concise, professional "
            "retention strategy from supplied customer data. Never invent customer facts, "
            "discounts, prices, service problems, or policies. Treat churn probability as a "
            "model estimate, not a certainty. Do not predict churn yourself. Use the supplied "
            "rule-based recommendation as the primary action guidance. Return exactly these "
            "sections: Risk Summary, Recommended Retention Strategy, Customer Message, and "
            "Agent Talking Points. Keep the customer message professional and concise."
        )
        customer_context = {
            "customer_id": str(customer.get("Customer_ID", "")),
            "state": str(customer.get("State", "")),
            "contract": str(customer.get("Contract", "")),
            "tenure_months": str(customer.get("Tenure_in_Months", "")),
            "monthly_charge": str(customer.get("Monthly_Charge", "")),
            "internet_type": str(customer.get("Internet_Type", "")),
            "premium_support": str(customer.get("Premium_Support", "")),
            "churn_probability": f"{float(customer['Churn_Probability']):.4f}",
            "risk_level": str(customer.get("Risk_Level", "")),
            "retention_priority": str(customer.get("Retention_Priority", "")),
            "rule_based_recommendation": str(customer.get("Retention_Recommendation", "")),
        }
        prompt = "Customer data:\n" + "\n".join(f"{k}: {v}" for k, v in customer_context.items())
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=gemini_model,
                contents=f"{system_prompt}\n\n{prompt}",
            )
            st.success(f"Gemini AI strategy generated successfully using {gemini_model}.")
            st.markdown(response.text)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
else:
    st.warning(
        "GenAI is not connected. Set GEMINI_API_KEY in .env and optionally GEMINI_MODEL "
        "to enable the AI strategy button."
    )
    st.markdown("**Rule-based strategy:** " + str(customer["Retention_Recommendation"]))

st.divider()
st.subheader("📊 Retention Priority List")
show_columns = [
    "Customer_ID", "Churn_Probability_Percent", "Risk_Level", "Retention_Priority",
    "Contract", "Tenure_in_Months", "Monthly_Charge", "Retention_Recommendation"
]
show_columns = [c for c in show_columns if c in filtered.columns]
st.dataframe(
    filtered[show_columns].sort_values("Churn_Probability_Percent", ascending=False),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Random Forest predicts churn risk. Business rules recommend retention actions. "
    "Gemini 3.1 Pro personalizes the communication when configured."
)
