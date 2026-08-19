"""Streamlit Use Case 12: Telecom Churn + Retention AI."""

import json
import os
import re
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

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Predicted Churners", f"{len(filtered):,}")
with c2:
    st.metric("High-Risk Customers", f"{(filtered['Risk_Level'].astype(str) == 'High').sum():,}")
with c3:
    st.metric("High-Priority Customers", f"{(filtered['Retention_Priority'].astype(str) == 'High').sum():,}")
with c4:
    st.metric("Average Churn Probability", f"{pd.to_numeric(filtered['Churn_Probability'], errors='coerce').mean():.1%}")

st.divider()

customer_ids = filtered["Customer_ID"].astype(str).tolist()
selected_id = st.selectbox("Select a customer", customer_ids)
customer = filtered[filtered["Customer_ID"].astype(str) == selected_id].iloc[0]

left, right = st.columns([1, 1])
with left:
    st.subheader("Customer Risk")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Churn Probability", f"{float(customer['Churn_Probability_Percent']):.1f}%")
    with r2:
        st.metric("Risk Level", str(customer["Risk_Level"]))
    st.write(f"**Retention Priority:** {customer['Retention_Priority']}")
    st.write(f"**Predicted Churn:** {'Yes' if int(customer['Predicted_Churn']) == 1 else 'No'}")

with right:
    st.subheader("Customer Profile")
    profile = {
        str(c).replace("_", " "): str(customer[c])
        for c in filtered.columns if c != "Customer_ID"
    }
    profile_df = pd.DataFrame(list(profile.items()), columns=["Attribute", "Value"])
    st.dataframe(profile_df, width="stretch", hide_index=True)

st.subheader("🎯 Retention Recommendation")
st.info(str(customer["Retention_Recommendation"]))

st.subheader("🤖 AI Retention Strategy")
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if gemini_key and genai is not None:
    if st.button("Generate Personalized AI Strategy", type="primary"):
        strategy_prompt = f"""
You are a telecom customer-retention assistant. Use ONLY the supplied customer data.
Do not invent discounts, prices, problems, policies, or customer facts. Treat churn probability
as a model estimate, not a certainty. Use the supplied rule-based recommendation as the action
baseline. Do not expose Customer_ID in the customer-facing message.
Return exactly: Risk Summary, Recommended Retention Strategy, Customer Message, Agent Talking Points.
Keep the language professional and simple.

Customer data:
{json.dumps({k: str(customer[k]) for k in customer.index}, default=str)}
"""
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model=gemini_model, contents=strategy_prompt)
            st.success(f"Gemini AI strategy generated successfully using {gemini_model}.")
            st.markdown(response.text)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")
else:
    st.warning("GenAI is not connected. Set GEMINI_API_KEY in .env.")
    st.markdown("**Rule-based strategy:** " + str(customer["Retention_Recommendation"]))

st.divider()

# ---------- Dynamic actual-data analytics ----------

def column_profile(data):
    result = {}
    for col in data.columns:
        s = data[col]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            vals = numeric.dropna()
            if len(vals):
                result[col] = {
                    "type": "numeric",
                    "non_null": int(vals.size),
                    "unique": int(s.nunique(dropna=True)),
                    "average": round(float(vals.mean()), 3),
                    "min": round(float(vals.min()), 3),
                    "max": round(float(vals.max()), 3),
                }
        else:
            counts = s.fillna("Unknown").astype(str).value_counts()
            result[col] = {
                "type": "categorical",
                "non_null": int(s.notna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in counts.head(15).items()},
            }
    return result


def churn_segment_insights(data):
    insights = {}
    churn = data[data["Predicted_Churn"].astype(str).isin(["1", "True", "true"])].copy()
    insights["total_rows"] = int(len(data))
    insights["predicted_churners"] = int(len(churn))
    insights["churn_rate_in_current_view"] = round(len(churn) / len(data), 4) if len(data) else 0
    for col in data.columns:
        if col in {"Customer_ID", "Retention_Recommendation"}:
            continue
        if data[col].dtype == "object" or data[col].nunique(dropna=True) <= 20:
            all_counts = data[col].fillna("Unknown").astype(str).value_counts()
            churn_counts = churn[col].fillna("Unknown").astype(str).value_counts() if len(churn) else pd.Series(dtype=int)
            rows = []
            for value, total in all_counts.head(15).items():
                c = int(churn_counts.get(value, 0))
                rows.append({
                    "value": str(value),
                    "customers": int(total),
                    "predicted_churners": c,
                    "predicted_churn_rate": round(c / int(total), 4) if total else 0,
                })
            insights[col] = rows
    return insights


def find_customer_from_question(question, data):
    q = question.lower()
    ids = data["Customer_ID"].astype(str).tolist()
    for cid in ids:
        if cid.lower() in q:
            return data[data["Customer_ID"].astype(str) == cid].iloc[0]
    # Also support a customer id typed with spaces/punctuation removed.
    compact_q = re.sub(r"[^a-z0-9]", "", q)
    for cid in ids:
        if re.sub(r"[^a-z0-9]", "", cid.lower()) in compact_q:
            return data[data["Customer_ID"].astype(str) == cid].iloc[0]
    return None


st.subheader("💬 Ask Your Data")
st.caption(
    "Ask about any available column, customer row, prediction, segment, or business insight. "
    "Python calculates facts from the actual prediction dataset; Gemini explains them in simple English."
)

question = st.text_input(
    "Ask a question",
    placeholder="Example: How many male customers are predicted to churn?",
    key="data_question",
)

if st.button("Ask Gemini", type="secondary") and question.strip():
    if not gemini_key or genai is None:
        st.warning("GenAI is not connected. Set GEMINI_API_KEY in .env.")
    else:
        data = filtered.copy()
        profiles = column_profile(data)
        insights = churn_segment_insights(data)
        row = find_customer_from_question(question, data)
        row_context = None
        if row is not None:
            row_context = {str(k): str(v) for k, v in row.items()}

        # Compact actual-data context. No fabricated values are supplied to the LLM.
        actual_context = {
            "current_filtered_rows": int(len(data)),
            "columns_available": [str(c) for c in data.columns],
            "column_profiles": profiles,
            "prediction_insights": insights,
            "selected_customer_row": {str(k): str(v) for k, v in customer.items()},
            "customer_row_found_from_question": row_context,
        }

        qa_prompt = f"""
You are the AI Data & Retention Analyst for a telecom churn prediction application.
Answer the user's question using ONLY the actual-data context below.

CORE RULES:
1. Never invent or estimate a number that is not present or directly calculable from the supplied data.
2. The current CSV is the prediction/retention dataset. Explain that results refer to the current filtered view.
3. You may answer questions about ANY available column listed in columns_available.
4. You may answer customer/row questions when the matching row is supplied.
5. For a category, use prediction_insights[<column>] where possible. It contains total customers,
   predicted churners, and predicted churn rate for each value.
6. For numeric columns, use column_profiles for average/min/max.
7. For questions asking "which", compare the supplied actual values and identify the highest/lowest.
8. For questions asking "why", use the customer's actual fields and retention recommendation. Do not claim causation;
   say "the model/rule engine identifies this as a signal".
9. Do not expose Customer_ID unless the user explicitly asks for it.
10. Use simple English. Give the direct answer first, then 1-3 useful insights.
11. If the requested information is not available, say exactly which column/data is missing.
12. Never claim that correlation or a feature caused churn unless the supplied data explicitly proves it.

USER QUESTION:
{question}

ACTUAL DATA CONTEXT:
{json.dumps(actual_context, default=str)}
"""
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model=gemini_model, contents=qa_prompt)
            st.success("Answer generated from the actual filtered prediction data.")
            st.markdown(response.text)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")

# Automatic business insights from the actual current filtered data.
st.divider()
st.subheader("📈 AI-Ready Data Insights")
st.caption("These insights are calculated directly from the current filtered prediction dataset.")

churn = filtered[filtered["Predicted_Churn"].astype(str).isin(["1", "True", "true"])].copy()
prob = pd.to_numeric(filtered["Churn_Probability"], errors="coerce")
ins1, ins2, ins3 = st.columns(3)
with ins1:
    st.write("**Current prediction view**")
    st.write(f"{len(filtered):,} customers are in the current filtered view, with {len(churn):,} predicted churners.")
with ins2:
    st.write("**Risk concentration**")
    high = int((filtered["Risk_Level"].astype(str) == "High").sum())
    st.write(f"{high:,} customers are classified as High Risk in this view.")
with ins3:
    st.write("**Average model probability**")
    st.write(f"{prob.mean():.1%} across the current filtered view.")

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
    "Gemini 2.5 Flash personalizes communication and explains actual data insights."
)
