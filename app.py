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


retention_df = load_csv(RETENTION_FILE)
full_df = load_csv(FULL_PREDICTION_FILE)
raw_df = load_csv(RAW_FILE)

if retention_df.empty:
    st.error("Retention_Recommendations.csv was not found.")
    st.code("python notebooks\\retention_recommendation.py")
    st.stop()

if full_df.empty:
    st.warning("Complete prediction data is missing. Run python notebooks\\generate_full_predictions.py for full-data analytics.")

required_columns = {
    "Customer_ID", "Churn_Probability", "Churn_Probability_Percent",
    "Predicted_Churn", "Risk_Level", "Retention_Priority", "Retention_Recommendation",
}
missing = required_columns.difference(retention_df.columns)
if missing:
    st.error(f"Missing columns in retention data: {', '.join(sorted(missing))}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS — applied to dashboard and AI context
# -----------------------------------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")
filtered = full_df.copy() if not full_df.empty else retention_df.copy()
filter_columns = ["Risk_Level", "Retention_Priority", "Contract", "State", "Internet_Type"]
for column in filter_columns:
    if column in filtered.columns:
        values = sorted(filtered[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(column.replace("_", " "), values, default=values, key=f"filter_{column}")
        filtered = filtered[filtered[column].astype(str).isin(selected)]

if filtered.empty:
    st.warning("No customers match the selected dashboard filters.")
    st.stop()

# -----------------------------------------------------------------------------
# EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
st.subheader("📊 Executive Overview")
prob = pd.to_numeric(filtered.get("Churn_Probability"), errors="coerce")
pred = pd.to_numeric(filtered.get("Predicted_Churn"), errors="coerce").fillna(0)
predicted_churners = int((pred == 1).sum())
total_customers = len(filtered)
churn_rate = predicted_churners / total_customers if total_customers else 0
avg_probability = float(prob.mean()) if prob.notna().any() else 0
high_risk = int((filtered.get("Risk_Level", pd.Series(dtype=str)).astype(str) == "High").sum())
high_priority = int((filtered.get("Retention_Priority", pd.Series(dtype=str)).astype(str) == "High").sum())

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Customers", f"{total_customers:,}")
with k2:
    st.metric("Predicted Churners", f"{predicted_churners:,}")
with k3:
    st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
with k4:
    st.metric("High-Risk", f"{high_risk:,}")
with k5:
    st.metric("Avg Churn Probability", f"{avg_probability:.1%}")

st.caption("These are model predictions, not confirmed future customer behavior.")

# -----------------------------------------------------------------------------
# POWER BI STYLE ANALYTICS
# -----------------------------------------------------------------------------
def segment_table(data, column):
    if column not in data.columns:
        return pd.DataFrame()
    out = data.copy()
    out["_pred"] = pd.to_numeric(out["Predicted_Churn"], errors="coerce").fillna(0)
    result = out.groupby(column, dropna=False).agg(
        Customers=("Customer_ID", "count"),
        Predicted_Churners=("_pred", "sum"),
        Avg_Churn_Probability=("Churn_Probability", "mean"),
    ).reset_index()
    result["Predicted_Churn_Rate"] = result["Predicted_Churners"] / result["Customers"]
    result["Predicted_Churners"] = result["Predicted_Churners"].astype(int)
    result["Avg_Churn_Probability"] = result["Avg_Churn_Probability"].round(4)
    result["Predicted_Churn_Rate"] = result["Predicted_Churn_Rate"].round(4)
    return result.sort_values("Predicted_Churn_Rate", ascending=False)

st.subheader("📈 Churn Analysis")
chart_tabs = st.tabs(["Contract", "State", "Internet Type", "Gender", "Risk", "Priority"])

with chart_tabs[0]:
    t = segment_table(filtered, "Contract")
    if not t.empty:
        st.bar_chart(t.set_index("Contract")["Predicted_Churn_Rate"])
        st.dataframe(t, width="stretch", hide_index=True)
with chart_tabs[1]:
    t = segment_table(filtered, "State")
    if not t.empty:
        st.bar_chart(t.head(15).set_index("State")["Predicted_Churn_Rate"])
        st.dataframe(t, width="stretch", hide_index=True)
with chart_tabs[2]:
    t = segment_table(filtered, "Internet_Type")
    if not t.empty:
        st.bar_chart(t.set_index("Internet_Type")["Predicted_Churn_Rate"])
        st.dataframe(t, width="stretch", hide_index=True)
with chart_tabs[3]:
    t = segment_table(filtered, "Gender")
    if not t.empty:
        st.bar_chart(t.set_index("Gender")["Predicted_Churn_Rate"])
        st.dataframe(t, width="stretch", hide_index=True)
with chart_tabs[4]:
    t = segment_table(filtered, "Risk_Level")
    if not t.empty:
        st.bar_chart(t.set_index("Risk_Level")["Customers"])
        st.dataframe(t, width="stretch", hide_index=True)
with chart_tabs[5]:
    t = segment_table(filtered, "Retention_Priority")
    if not t.empty:
        st.bar_chart(t.set_index("Retention_Priority")["Customers"])
        st.dataframe(t, width="stretch", hide_index=True)

st.subheader("🎯 Retention Insights")
risk_col, priority_col = st.columns(2)
with risk_col:
    risk_counts = filtered["Risk_Level"].astype(str).value_counts() if "Risk_Level" in filtered else pd.Series(dtype=int)
    st.write("**Risk Distribution**")
    st.dataframe(risk_counts.rename("Customers").reset_index().rename(columns={"index": "Risk_Level"}), width="stretch", hide_index=True)
with priority_col:
    priority_counts = filtered["Retention_Priority"].astype(str).value_counts() if "Retention_Priority" in filtered else pd.Series(dtype=int)
    st.write("**Retention Priority**")
    st.dataframe(priority_counts.rename("Customers").reset_index().rename(columns={"index": "Retention_Priority"}), width="stretch", hide_index=True)

# -----------------------------------------------------------------------------
# CUSTOMER 360
# -----------------------------------------------------------------------------
st.subheader("👤 Customer 360")
customer_source = filtered[filtered["Predicted_Churn"].astype(int) == 1].copy()
if customer_source.empty:
    customer_source = filtered.copy()
customer_ids = customer_source["Customer_ID"].astype(str).tolist()
selected_id = st.selectbox("Select a customer", customer_ids)
customer = customer_source[customer_source["Customer_ID"].astype(str) == selected_id].iloc[0]

left, right = st.columns([1, 1])
with left:
    st.write("**Customer Risk**")
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Churn Probability", f"{float(customer['Churn_Probability_Percent']):.1f}%")
    with r2:
        st.metric("Risk Level", str(customer["Risk_Level"]))
    st.write(f"**Retention Priority:** {customer['Retention_Priority']}")
    st.write(f"**Predicted Churn:** {'Yes' if int(customer['Predicted_Churn']) == 1 else 'No'}")
with right:
    st.write("**Customer Profile**")
    profile = {str(c).replace("_", " "): str(customer[c]) for c in customer.index if c != "Customer_ID"}
    st.dataframe(pd.DataFrame(list(profile.items()), columns=["Attribute", "Value"]), width="stretch", hide_index=True)

st.write("**🎯 Retention Recommendation**")
st.info(str(customer["Retention_Recommendation"]))

# -----------------------------------------------------------------------------
# GEMINI
# -----------------------------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.subheader("🤖 AI Retention Strategy")
if gemini_key and genai is not None:
    if st.button("Generate Personalized AI Strategy", type="primary"):
        strategy_prompt = f"""
You are a telecom customer-retention assistant. Use ONLY the supplied customer data.
Never invent discounts, prices, problems, policies, or customer facts. Treat churn probability as a model estimate.
Use the rule-based recommendation as the action baseline. Do not expose Customer_ID in the customer-facing message.
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

# -----------------------------------------------------------------------------
# DATA-DRIVEN GEMINI ANALYST
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


def build_context(data):
    """Calculate factual dashboard context in Python before Gemini sees it."""
    context = {"total_customers": int(len(data)), "columns_available": [str(c) for c in data.columns]}
    pred = pd.to_numeric(data["Predicted_Churn"], errors="coerce").fillna(0)
    churn = data[pred == 1]
    context["predicted_churners"] = int(len(churn))
    context["predicted_churn_rate"] = round(len(churn) / len(data), 4) if len(data) else 0
    context["average_churn_probability"] = round(float(pd.to_numeric(data["Churn_Probability"], errors="coerce").mean()), 4)

    for col in data.columns:
        if col in {"Customer_ID", "Retention_Recommendation"}:
            continue
        numeric = pd.to_numeric(data[col], errors="coerce")
        if numeric.notna().mean() >= 0.8:
            vals = numeric.dropna()
            context.setdefault("numeric", {})[col] = {
                "average": round(float(vals.mean()), 3) if len(vals) else None,
                "minimum": round(float(vals.min()), 3) if len(vals) else None,
                "maximum": round(float(vals.max()), 3) if len(vals) else None,
            }
        else:
            counts = data[col].fillna("Unknown").astype(str).value_counts()
            total_counts = counts.to_dict()
            churn_counts = churn[col].fillna("Unknown").astype(str).value_counts().to_dict() if col in churn.columns else {}
            segments = {}
            for value, total in list(total_counts.items())[:40]:
                c = int(churn_counts.get(value, 0))
                segments[str(value)] = {
                    "customers": int(total),
                    "predicted_churners": c,
                    "predicted_churn_rate": round(c / total, 4) if total else 0,
                }
            context.setdefault("categorical", {})[col] = segments

    # Top segment findings for business insight generation.
    top_segments = []
    for col in context.get("categorical", {}):
        segs = context["categorical"][col]
        eligible = [(v, d) for v, d in segs.items() if d["customers"] >= max(10, int(len(data) * 0.01))]
        if eligible:
            value, stats = max(eligible, key=lambda x: x[1]["predicted_churn_rate"])
            top_segments.append({"column": col, "value": value, **stats})
    context["highest_risk_segments"] = sorted(top_segments, key=lambda x: x["predicted_churn_rate"], reverse=True)[:10]
    return context


st.divider()
st.subheader("💬 AI Data Analyst")
st.caption("Power BI-style analytics with natural-language questions. Dashboard filters apply to the data used for this analysis.")

if not full_df.empty:
    st.success(f"Connected to {len(full_df):,} complete customer prediction rows.")

q_col, button_col = st.columns([5, 1])
with q_col:
    question = st.text_input("Ask about the dashboard, data, predictions or a customer", placeholder="e.g. Which contract has the highest predicted churn rate?", key="analyst_question")
with button_col:
    ask = st.button("Ask AI", type="primary")

if ask and question.strip():
    if not gemini_key or genai is None:
        st.warning("GenAI is not connected. Set GEMINI_API_KEY in .env.")
    else:
        data_for_ai = filtered.copy()
        context = build_context(data_for_ai)
        row = find_customer(question, data_for_ai)
        row_context = {str(k): str(v) for k, v in row.items()} if row is not None else None
        prompt = f"""
You are the AI Data Analyst for a telecom Power BI-style churn dashboard.
Answer in simple English for a business user.

The data context below was calculated directly from the CURRENT DASHBOARD FILTERED DATA by Python.
Use ONLY these facts. Never invent numbers.

Rules:
1. Give the direct answer first.
2. Then provide up to 3 useful insights.
3. For "highest/lowest", compare the supplied segment churn rates.
4. For counts, use supplied customer/churner counts.
5. For averages, use supplied Python averages.
6. If a column is not available, say so.
7. If a customer row is supplied, answer customer-specific questions from that row.
8. Do not claim correlation or causation unless the data explicitly establishes it. Use "associated with" or "model signal".
9. Clearly distinguish predicted churn from confirmed historical churn.
10. Do not expose Customer_ID unless explicitly requested.
11. For broad questions like "give me insights", provide 3-5 concise, actionable insights and mention the supporting numbers.
12. For retention questions, use the available retention recommendation/risk fields; do not invent offers.

USER QUESTION:
{question}

CURRENT DASHBOARD DATA:
{json.dumps(context, default=str)}

MATCHING CUSTOMER ROW, IF FOUND:
{json.dumps(row_context, default=str)}
"""
        try:
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model=gemini_model, contents=prompt)
            st.success(f"Analysis generated from the current dashboard data using {gemini_model}.")
            st.markdown(response.text)
        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")

# -----------------------------------------------------------------------------
# RETENTION PRIORITY TABLE
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📊 Retention Priority List")
retention_view = filtered[filtered["Predicted_Churn"].astype(int) == 1].copy()
if retention_view.empty:
    retention_view = filtered.copy()
show_columns = [
    "Customer_ID", "Churn_Probability_Percent", "Risk_Level", "Retention_Priority",
    "Contract", "Tenure_in_Months", "Monthly_Charge", "Retention_Recommendation"
]
show_columns = [c for c in show_columns if c in retention_view.columns]
st.dataframe(
    retention_view[show_columns].sort_values("Churn_Probability_Percent", ascending=False),
    width="stretch",
    hide_index=True,
)
st.caption("Random Forest predicts churn risk for all customers. The retention view prioritizes predicted churners. Gemini explains dashboard data and personalizes retention communication.")
