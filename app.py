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
RETENTION_FILE = ROOT / "data" / "predictions" / "Retention_Recommendations.csv"
FULL_PREDICTION_FILE = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"
RAW_FILE = ROOT / "data" / "raw" / "Customer_Data.csv"
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Telecom Retention AI", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Retention AI")
st.caption("Use Case 12 — Churn Prediction + Personalized Retention Recommendation")


@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# Retention dashboard remains focused on predicted churners.
retention_df = load_csv(RETENTION_FILE)
# Q&A uses the complete prediction dataset so it can answer about all customers.
full_df = load_csv(FULL_PREDICTION_FILE)
raw_df = load_csv(RAW_FILE)

if retention_df.empty:
    st.error("Retention_Recommendations.csv was not found.")
    st.code("python notebooks\\retention_recommendation.py")
    st.stop()

if full_df.empty:
    st.warning(
        "The complete prediction dataset is not generated yet. Run "
        "python notebooks\\generate_full_predictions.py so Ask Your Data can use every customer row."
    )

required_columns = {
    "Customer_ID", "Churn_Probability", "Churn_Probability_Percent",
    "Predicted_Churn", "Risk_Level", "Retention_Priority",
    "Retention_Recommendation",
}
missing = required_columns.difference(retention_df.columns)
if missing:
    st.error(f"Missing columns in retention data: {', '.join(sorted(missing))}")
    st.stop()

st.sidebar.header("Customer Filters")
filtered = retention_df.copy()
for column in ["Risk_Level", "Retention_Priority", "Contract", "State", "Internet_Type"]:
    if column in filtered.columns:
        values = sorted(filtered[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(column.replace("_", " "), values, default=values)
        filtered = filtered[filtered[column].astype(str).isin(selected)]

if filtered.empty:
    st.warning("No predicted-churn customers match the selected filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Predicted Churners", f"{len(filtered):,}")
with c2:
    st.metric("High-Risk Customers", f"{(filtered['Risk_Level'].astype(str) == 'High').sum():,}")
with c3:
    st.metric("High-Priority Customers", f"{(filtered['Retention_Priority'].astype(str) == 'High').sum():,}")
with c4:
    p = pd.to_numeric(filtered["Churn_Probability"], errors="coerce")
    st.metric("Average Churn Probability", f"{p.mean():.1%}")

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
    st.dataframe(pd.DataFrame(list(profile.items()), columns=["Attribute", "Value"]), width="stretch", hide_index=True)

st.subheader("🎯 Retention Recommendation")
st.info(str(customer["Retention_Recommendation"]))

st.subheader("🤖 AI Retention Strategy")
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if gemini_key and genai is not None:
    if st.button("Generate Personalized AI Strategy", type="primary"):
        strategy_prompt = f"""
You are a telecom customer-retention assistant. Use ONLY the supplied customer data.
Never invent discounts, prices, problems, policies, or customer facts. Treat churn probability
as a model estimate, not a certainty. Use the supplied rule-based recommendation as the action baseline.
Do not expose Customer_ID in the customer-facing message.
Return exactly: Risk Summary, Recommended Retention Strategy, Customer Message, Agent Talking Points.
Use simple professional English.

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

st.divider()

# -----------------------------------------------------------------------------
# AI DATA & RETENTION ASSISTANT
# -----------------------------------------------------------------------------

def find_customer(question, data):
    if data.empty or "Customer_ID" not in data.columns:
        return None
    q = question.lower()
    for cid in data["Customer_ID"].astype(str):
        if cid.lower() in q:
            return data[data["Customer_ID"].astype(str) == cid].iloc[0]
    compact = re.sub(r"[^a-z0-9]", "", q)
    for cid in data["Customer_ID"].astype(str):
        if re.sub(r"[^a-z0-9]", "", cid.lower()) in compact:
            return data[data["Customer_ID"].astype(str) == cid].iloc[0]
    return None


def build_actual_context(data):
    """Create compact facts from the complete prediction dataset.

    Python performs the calculations. Gemini only explains the resulting facts.
    """
    context = {
        "total_customers": int(len(data)),
        "columns_available": [str(c) for c in data.columns],
    }

    if "Predicted_Churn" in data.columns:
        pred = pd.to_numeric(data["Predicted_Churn"], errors="coerce").fillna(0)
        churn = data[pred == 1]
        context["predicted_churners"] = int(len(churn))
        context["predicted_churn_rate"] = round(float(len(churn) / len(data)), 4) if len(data) else 0
    else:
        churn = pd.DataFrame()

    if "Risk_Level" in data.columns:
        context["risk_counts"] = data["Risk_Level"].fillna("Unknown").astype(str).value_counts().to_dict()
    if "Retention_Priority" in data.columns:
        context["priority_counts"] = data["Retention_Priority"].fillna("Unknown").astype(str).value_counts().to_dict()

    # Every available column is represented. This makes Q&A dynamic rather than hard-coded.
    column_facts = {}
    for col in data.columns:
        s = data[col]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            values = numeric.dropna()
            if len(values):
                column_facts[col] = {
                    "type": "numeric",
                    "non_null": int(len(values)),
                    "unique": int(s.nunique(dropna=True)),
                    "average": round(float(values.mean()), 3),
                    "minimum": round(float(values.min()), 3),
                    "maximum": round(float(values.max()), 3),
                }
        else:
            counts = s.fillna("Unknown").astype(str).value_counts()
            column_facts[col] = {
                "type": "categorical",
                "non_null": int(s.notna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "values": {str(k): int(v) for k, v in counts.head(30).items()},
            }

            # For categorical columns, provide actual churn rate for every value.
            if len(churn) and col in churn.columns:
                churn_counts = churn[col].fillna("Unknown").astype(str).value_counts()
                total_counts = s.fillna("Unknown").astype(str).value_counts()
                segment = {}
                for value, total in total_counts.head(30).items():
                    c = int(churn_counts.get(value, 0))
                    segment[str(value)] = {
                        "customers": int(total),
                        "predicted_churners": c,
                        "predicted_churn_rate": round(c / int(total), 4) if total else 0,
                    }
                column_facts[col]["prediction_by_value"] = segment

    context["columns"] = column_facts

    # Useful numeric relationships calculated directly by Python.
    numeric_cols = []
    for col in data.columns:
        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().mean() >= 0.8:
            numeric_cols.append(col)
    if numeric_cols and "Predicted_Churn" in data.columns:
        comparison = {}
        pred = pd.to_numeric(data["Predicted_Churn"], errors="coerce").fillna(0)
        for col in numeric_cols:
            values = pd.to_numeric(data[col], errors="coerce")
            comparison[col] = {
                "average_all_customers": round(float(values.mean()), 3) if values.notna().any() else None,
                "average_predicted_churners": round(float(values[pred == 1].mean()), 3) if (pred == 1).any() else None,
            }
        context["numeric_comparison_churners_vs_all"] = comparison

    return context


st.subheader("💬 Ask Your Data")
if not full_df.empty:
    st.success(f"AI Q&A is connected to the complete prediction dataset: {len(full_df):,} customer rows.")
else:
    st.warning("Run the full prediction script first. Q&A is not yet using the complete customer dataset.")

st.caption(
    "Ask about any column, any customer row, predictions, segments, comparisons, or retention insights. "
    "Python calculates facts from the complete prediction dataset; Gemini explains them in simple English."
)

question = st.text_input(
    "Ask a question",
    placeholder="Example: How many male customers are predicted to churn?",
    key="data_question",
)

if st.button("Ask Gemini", type="secondary") and question.strip():
    if not gemini_key or genai is None:
        st.warning("GenAI is not connected. Set GEMINI_API_KEY in .env.")
    elif full_df.empty:
        st.error("Complete prediction data is missing. Run generate_full_predictions.py first.")
    else:
        qa_data = full_df.copy()
        actual_context = build_actual_context(qa_data)
        row = find_customer(question, qa_data)
        row_context = {str(k): str(v) for k, v in row.items()} if row is not None else None

        prompt = f"""
You are an AI Data & Retention Analyst for a telecom churn prediction project.
Answer the user's question in SIMPLE ENGLISH using ONLY the actual-data context below.

IMPORTANT:
- The complete prediction dataset contains ALL customer rows, not just predicted churners.
- Answer from the full dataset unless the user explicitly asks about the dashboard's filtered retention view.
- Never invent a number, row, category, or business fact.
- Python calculated all counts, rates, averages, minimums and maximums in the context.
- For categorical questions, use columns.<column>.prediction_by_value when asking about churn rate by category.
- For questions such as "how many male customers are predicted to churn", use the Gender prediction_by_value entry.
- For questions such as "which state has the highest churn rate", compare prediction_by_value rates.
- For numeric comparisons, use numeric_comparison_churners_vs_all.
- For customer questions, use the matching customer row supplied below.
- If a requested column does not exist, say that the column is not available.
- If a requested row/customer does not exist, say it was not found.
- Do not claim a feature caused churn. Say it is a model-associated signal when appropriate.
- Do not expose Customer_ID unless the user explicitly asks for it.
- Give the direct answer first. Then give 1-3 useful insights.
- If the user asks for a list, provide a concise list/table based only on the supplied facts.

USER QUESTION:
{question}

COMPLETE ACTUAL PREDICTION DATA FACTS:
{json.dumps(actual_context, default=str)}

MATCHING CUSTOMER ROW, IF FOUND:
{json.dumps(row_context, default=str)}
"""
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model=gemini_model, contents=prompt)
            st.success("Answer generated from the complete customer prediction dataset.")
            st.markdown(response.text)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")

st.divider()
st.subheader("📈 Full-Data Prediction Insights")
if not full_df.empty:
    total = len(full_df)
    pred = pd.to_numeric(full_df["Predicted_Churn"], errors="coerce").fillna(0)
    churn_count = int((pred == 1).sum())
    avg_prob = pd.to_numeric(full_df["Churn_Probability"], errors="coerce").mean()
    a, b, c, d = st.columns(4)
    with a:
        st.metric("All Customer Rows", f"{total:,}")
    with b:
        st.metric("Predicted Churners", f"{churn_count:,}")
    with c:
        st.metric("Predicted Churn Rate", f"{churn_count / total:.1%}")
    with d:
        st.metric("Average Churn Probability", f"{avg_prob:.1%}")

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
    "Random Forest predicts churn risk for all customers. The retention dashboard focuses on predicted churners. "
    "Gemini 2.5 Flash explains the complete prediction dataset and personalizes retention communication."
)
