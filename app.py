"""Streamlit Use Case 12: Telecom Churn + Retention AI."""
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

ROOT = Path(__file__).resolve().parent
RETENTION_FILE = ROOT / "data" / "predictions" / "Retention_Recommendations.csv"
FULL_FILE = ROOT / "data" / "predictions" / "All_Customer_Predictions.csv"
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Telecom Customer Retention AI", page_icon="📡", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


retention_df = load_data(RETENTION_FILE)
full_df = load_data(FULL_FILE)
if full_df.empty:
    full_df = retention_df.copy()
if full_df.empty:
    st.error("Prediction data is missing. Run: python notebooks\\retention_recommendation.py")
    st.stop()


# ----------------------------- Helpers --------------------------------------
def prediction_metrics(df):
    pred = pd.to_numeric(df["Predicted_Churn"], errors="coerce").fillna(0)
    prob = pd.to_numeric(df["Churn_Probability"], errors="coerce")
    n = len(df)
    churners = int((pred == 1).sum())
    return {
        "customers": n,
        "predicted_churners": churners,
        "predicted_churn_rate": churners / n if n else 0,
        "avg_probability": float(prob.mean()) if prob.notna().any() else 0,
        "high_risk": int((df["Risk_Level"].astype(str) == "High").sum()),
        "high_priority": int((df["Retention_Priority"].astype(str) == "High").sum()),
    }


def segment_table(df, column):
    if column not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["_pred"] = pd.to_numeric(x["Predicted_Churn"], errors="coerce").fillna(0)
    x["_prob"] = pd.to_numeric(x["Churn_Probability"], errors="coerce")
    out = x.groupby(column, dropna=False).agg(
        Customers=("Customer_ID", "count"),
        Predicted_Churners=("_pred", "sum"),
        Avg_Churn_Probability=("_prob", "mean"),
    ).reset_index()
    out["Predicted_Churners"] = out["Predicted_Churners"].astype(int)
    out["Predicted_Churn_Rate"] = out["Predicted_Churners"] / out["Customers"]
    return out.sort_values("Predicted_Churn_Rate", ascending=False)


def find_customer(question, df):
    if "Customer_ID" not in df.columns:
        return None
    q = question.lower()
    ids = df["Customer_ID"].astype(str)
    for cid in ids:
        if cid.lower() in q:
            return df[ids == cid].iloc[0]
    compact = re.sub(r"[^a-z0-9]", "", q)
    for cid in ids:
        normalized = re.sub(r"[^a-z0-9]", "", cid.lower())
        if normalized and normalized in compact:
            return df[ids == cid].iloc[0]
    return None


def get_secret(name, default=None):
    """Read Streamlit Cloud secrets first, then local environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def groq_call(prompt):
    """Make one fresh Groq request per Streamlit interaction.

    Python remains the source of truth for all dashboard calculations.
    Groq is used only to understand questions and explain verified results.
    """
    key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")

    if not key:
        raise RuntimeError(
            "Groq is not connected. Add GROQ_API_KEY to Streamlit Secrets or .env"
        )

    if Groq is None:
        raise RuntimeError(
            "groq is not installed. Run: pip install groq"
        )

    client = Groq(api_key=key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a telecom business intelligence assistant. "
                    "Use only the supplied verified Python result. "
                    "Never recalculate, change, or invent numbers. "
                    "Use simple professional English. "
                    "Clearly say predicted churn, not confirmed churn."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=900,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned an empty response.")

    return text.strip()



# ----------------------------- Filters --------------------------------------
st.sidebar.header("🔎 Dashboard Filters")
filtered = full_df.copy()
for col in ["Risk_Level", "Retention_Priority", "Contract", "State", "Internet_Type"]:
    if col in filtered.columns:
        vals = sorted(filtered[col].dropna().astype(str).unique())
        chosen = st.sidebar.multiselect(col.replace("_", " "), vals, default=vals, key="f_" + col)
        filtered = filtered[filtered[col].astype(str).isin(chosen)]

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


# ----------------------------- Dashboard ------------------------------------
st.title("📡 Telecom Customer Retention AI")
st.caption("Use Case 12 — Churn Prediction + Personalized Retention Recommendation")

m = prediction_metrics(filtered)
st.subheader("📊 Executive Overview")
a, b, c, d, e = st.columns(5)
a.metric("Customers", f"{m['customers']:,}")
b.metric("Predicted Churners", f"{m['predicted_churners']:,}")
c.metric("Predicted Churn Rate", f"{m['predicted_churn_rate']:.1%}")
d.metric("High-Risk", f"{m['high_risk']:,}")
e.metric("Avg Churn Probability", f"{m['avg_probability']:.1%}")
st.caption("Predicted churn is a model estimate, not confirmed future customer behavior.")

def format_pct(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def build_category_analysis(df, column):
    """Build industry-style count, share and churn-rate metrics."""
    if column not in df.columns:
        return pd.DataFrame()

    x = df.copy()

    x["_pred"] = pd.to_numeric(
        x["Predicted_Churn"],
        errors="coerce",
    ).fillna(0)

    x["_prob"] = pd.to_numeric(
        x["Churn_Probability"],
        errors="coerce",
    )

    total_customers = len(x)

    out = (
        x.groupby(column, dropna=False)
        .agg(
            Customers=("Customer_ID", "count"),
            Predicted_Churners=("_pred", "sum"),
            Avg_Churn_Probability=("_prob", "mean"),
        )
        .reset_index()
    )

    out["Predicted_Churners"] = (
        out["Predicted_Churners"].astype(int)
    )

    out["Customer_Share"] = (
        out["Customers"] / total_customers
        if total_customers
        else 0
    )

    out["Predicted_Churn_Rate"] = (
        out["Predicted_Churners"] / out["Customers"]
    ).fillna(0)

    total_churners = int(
        out["Predicted_Churners"].sum()
    )

    out["Churner_Share"] = (
        out["Predicted_Churners"] / total_churners
        if total_churners
        else 0
    )

    return out.sort_values(
        "Predicted_Churn_Rate",
        ascending=False,
    )


def show_category_analysis(df, column, title):
    """Display a business-friendly categorical churn analysis."""
    t = build_category_analysis(
        df,
        column,
    )

    if t.empty:
        st.info(
            f"{title} is not available in the prediction data."
        )
        return

    # Primary industry KPI: compare segments by churn rate,
    # not raw churner count, because segment sizes differ.
    st.markdown(
        f"**{title} — Predicted Churn Rate (%)**"
    )

    chart_df = t[
        [column, "Predicted_Churn_Rate"]
    ].copy()

    chart_df["Predicted Churn Rate (%)"] = (
        chart_df["Predicted_Churn_Rate"] * 100
    )

    chart_df = chart_df.drop(
        columns=["Predicted_Churn_Rate"]
    )

    # Horizontal bars are easier to read for many categorical labels.
    st.bar_chart(
        chart_df.set_index(column),
        y="Predicted Churn Rate (%)",
    )

    top = t.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Highest-Risk Segment",
        str(top[column]),
    )

    c2.metric(
        "Highest Churn Rate",
        format_pct(top["Predicted_Churn_Rate"]),
    )

    c3.metric(
        "Predicted Churners",
        f"{int(t['Predicted_Churners'].sum()):,}",
    )

    # Keep counts for operational context, but make percentages
    # the primary comparison metrics.
    show = t.rename(
        columns={
            column: title,
            "Customers": "Customers",
            "Predicted_Churners": "Predicted Churners",
        }
    ).copy()

    show["Customer Share"] = show[
        "Customer_Share"
    ].map(format_pct)

    show["Churner Share"] = show[
        "Churner_Share"
    ].map(format_pct)

    show["Predicted Churn Rate"] = show[
        "Predicted_Churn_Rate"
    ].map(format_pct)

    show["Avg Churn Probability"] = show[
        "Avg_Churn_Probability"
    ].map(format_pct)

    show = show[
        [
            title,
            "Customers",
            "Customer Share",
            "Predicted Churners",
            "Churner Share",
            "Predicted Churn Rate",
            "Avg Churn Probability",
        ]
    ]

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
    )


st.subheader("📈 Churn Analysis")

st.caption(
    "Industry-style BI analysis: use counts for volume, "
    "customer share for population mix, and predicted churn "
    "rate for fair segment-to-segment comparison."
)

tabs = st.tabs(
    [
        "Contract",
        "State",
        "Internet Type",
        "Gender",
        "Risk",
        "Priority",
    ]
)

for tab, column, title in zip(
    tabs,
    [
        "Contract",
        "State",
        "Internet_Type",
        "Gender",
        "Risk_Level",
        "Retention_Priority",
    ],
    [
        "Contract",
        "State",
        "Internet Type",
        "Gender",
        "Risk Level",
        "Retention Priority",
    ],
):
    with tab:
        show_category_analysis(
            filtered,
            column,
            title,
        )


st.subheader("👤 Customer 360")
cs = filtered[filtered["Predicted_Churn"].astype(int) == 1].copy()
if cs.empty:
    cs = filtered.copy()
selected = st.selectbox("Select a customer", cs["Customer_ID"].astype(str).tolist())
customer = cs[cs["Customer_ID"].astype(str) == selected].iloc[0]
left, right = st.columns(2)
with left:
    x, y = st.columns(2)
    x.metric("Churn Probability", f"{float(customer['Churn_Probability_Percent']):.1f}%")
    y.metric("Risk Level", str(customer["Risk_Level"]))
    st.write(f"**Retention Priority:** {customer['Retention_Priority']}")
    st.write(f"**Predicted Churn:** {'Yes' if int(customer['Predicted_Churn']) == 1 else 'No'}")
with right:
    profile = pd.DataFrame(
        [(c.replace("_", " "), str(customer[c])) for c in customer.index if c != "Customer_ID"],
        columns=["Attribute", "Value"],
    )
    st.dataframe(profile, width="stretch", hide_index=True)
st.write("**🎯 Retention Recommendation**")
st.info(str(customer.get("Retention_Recommendation", "No recommendation available.")))


# ----------------------------- Groq retention -----------------------------
st.subheader("🤖 AI Retention Strategy")
if st.button("Generate Personalized AI Strategy", type="primary"):
    try:
        prompt = f"""You are a telecom retention assistant. Use only this customer data. Treat churn probability as a model estimate. Never invent prices, discounts, policies or facts. Use the rule-based recommendation as the baseline. Use simple professional English. Return: Risk Summary, Recommended Retention Strategy, Customer Message, Agent Talking Points. Do not expose Customer_ID.

Customer data:
{json.dumps({k: str(customer[k]) for k in customer.index}, default=str)}"""
        st.markdown(groq_call(prompt))
    except Exception as exc:
        st.error(f"Groq request failed: {exc}")


# ----------------------------- AI Data Analyst -------------------------------
st.divider()
st.subheader("💬 AI Data Analyst")
st.caption(
    "Power BI-style natural-language analysis. "
    "Ask questions about customers, churn, risk, retention, "
    "and business performance."
)

question = st.text_input(
    "Ask Your Data",
    placeholder="Examples: What is total predicted churn? | Which state has highest churn rate? | How many female churners? | Average monthly charge? | Top 5 contracts | Tell me about customer 62359-AND",
    key="ai_question",
)
ask = st.button("🔍 Analyze", type="primary")


def get_plan(question, df):
    """Use Groq only to classify the question; Python calculates the answer."""
    columns = [str(c) for c in df.columns]
    prompt = f"""You are a BI query planner. Return JSON only. Available columns: {json.dumps(columns)}.
User question: {question}

Schema: {{"type":"overview|group|numeric|customer","group_by":null or exact categorical column,"target_column":null or exact numeric column,"metric":"customers|predicted_churners|predicted_churn_rate|average_churn_probability|average|min|max","top_n":10,"sort":"asc|desc"}}
Rules: churn/churners=Predicted_Churn; churn rate=Predicted_Churn_Rate; female/male/state/contract/internet/risk/priority questions use that exact column; average monthly charge uses target_column=Monthly_Charge and no group; top N sets top_n; Customer-ID questions are handled by Python before this planner. Never classify female/male/gender/state/contract/internet/risk/priority questions as customer questions. Do not invent columns."""
    text = groq_call(prompt).strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.I).strip()
    return json.loads(text)


def fallback_plan(question, df):
    q = question.lower()
    aliases = {
        "gender": "Gender",
        "state": "State",
        "contract": "Contract",
        "internet": "Internet_Type",
        "risk": "Risk_Level",
        "priority": "Retention_Priority",
        "payment": "Payment_Method",
        "married": "Married",
        "tenure": "Tenure_in_Months",
        "monthly charge": "Monthly_Charge",
        "monthly": "Monthly_Charge",
        "revenue": "Total_Revenue",
        "refund": "Total_Refunds",
        "referral": "Number_of_Referrals",
        "probability": "Churn_Probability",
    }
    group = next(
        (v for k, v in sorted(aliases.items(), key=lambda x: -len(x[0])) if k in q and v in df.columns and k not in ["monthly", "revenue", "refund", "referral", "probability"]),
        None,
    )
    target = next((v for k, v in sorted(aliases.items(), key=lambda x: -len(x[0])) if k in q and v in df.columns), None)
    nmatch = re.search(r"top\s+(\d+)", q)
    n = max(1, min(50, int(nmatch.group(1)))) if nmatch else 10
    if any(w in q for w in ["average", "mean"]):
        metric = "average"
    elif any(w in q for w in ["highest", "maximum", "max", "lowest", "minimum", "min"]):
        metric = "max" if any(w in q for w in ["highest", "maximum", "max"]) else "min"
    elif "rate" in q or "percentage" in q:
        metric = "predicted_churn_rate"
    elif "churn" in q:
        metric = "predicted_churners"
    else:
        metric = "customers"
    typ = "customer" if "customer " in q or "customer id" in q else ("group" if group else "numeric" if metric in ["average", "min", "max"] else "overview")
    return {"type": typ, "group_by": group, "target_column": target, "metric": metric, "top_n": n, "sort": "desc"}



def normalize_plan(plan, question, df):
    """Sanitize Groq's plan; deterministic rules override ambiguous intent."""
    if not isinstance(plan, dict):
        return fallback_plan(question, df)

    allowed_types = {"overview", "group", "numeric"}
    allowed_metrics = {
        "customers", "predicted_churners", "predicted_churn_rate",
        "average_churn_probability", "average", "min", "max"
    }

    typ = plan.get("type") if plan.get("type") in allowed_types else None
    group = plan.get("group_by")
    if not isinstance(group, str) or group not in df.columns:
        group = None
    target = plan.get("target_column")
    if not isinstance(target, str) or target not in df.columns:
        target = None
    metric = plan.get("metric") if plan.get("metric") in allowed_metrics else None

    raw_n = plan.get("top_n")
    try:
        n = int(raw_n) if raw_n is not None else 10
    except (TypeError, ValueError):
        n = 10
    n = max(1, min(50, n))

    sort = plan.get("sort") if plan.get("sort") in {"asc", "desc"} else "desc"
    q = question.lower()

    # Deterministic group intent. This prevents "female customer" from becoming Customer-360.
    for words, column in [
        (["female", "male", "gender"], "Gender"),
        (["state"], "State"),
        (["contract"], "Contract"),
        (["internet"], "Internet_Type"),
        (["risk"], "Risk_Level"),
        (["priority"], "Retention_Priority"),
    ]:
        if any(w in q for w in words) and column in df.columns:
            group = column
            typ = "group"
            break

    if "churn rate" in q or "churn percentage" in q or "churn percent" in q:
        metric = "predicted_churn_rate"
    elif any(x in q for x in ["predicted churners", "predicted churn", "churners", "likely to churn"]):
        metric = "predicted_churners"

    if "average monthly charge" in q or "average monthly" in q:
        metric = "average"
        target = "Monthly_Charge" if "Monthly_Charge" in df.columns else target
        typ = "numeric"
        group = None

    if group is not None:
        typ = "group"
    if typ is None:
        typ = "numeric" if metric in {"average", "min", "max"} else "overview"
    if metric is None:
        metric = "customers"

    return {"type": typ, "group_by": group, "target_column": target,
            "metric": metric, "top_n": n, "sort": sort}

def execute(plan, df, question_text):
    typ = plan.get("type", "overview")
    group = plan.get("group_by")
    target = plan.get("target_column")
    metric = plan.get("metric", "customers")

    # Groq can return top_n as null.
    # Never call int(None).
    raw_n = plan.get("top_n", 10)

    try:
        n = int(raw_n) if raw_n is not None else 10
    except (TypeError, ValueError):
        n = 10

    n = max(1, min(50, n))

    if typ == "customer":
        row = find_customer(question_text, df)
        return {"kind": "customer", "row": row}

    if group not in df.columns:
        group = None

    if group:
        t = segment_table(df, group)
        value = {
            "customers": "Customers",
            "predicted_churners": "Predicted_Churners",
            "predicted_churn_rate": "Predicted_Churn_Rate",
            "average_churn_probability": "Avg_Churn_Probability",
        }.get(metric, "Predicted_Churn_Rate")
        t = t.sort_values(value, ascending=(plan.get("sort") == "asc")).head(n)
        return {"kind": "group", "table": t, "group": group, "value": value}

    if metric in ["average", "min", "max"]:
        if target not in df.columns or pd.to_numeric(df[target], errors="coerce").notna().mean() < 0.8:
            numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
            target = numeric[0] if numeric else None
        if not target:
            return {"kind": "empty"}
        v = pd.to_numeric(df[target], errors="coerce").dropna()
        val = v.mean() if metric == "average" else v.min() if metric == "min" else v.max()
        return {"kind": "numeric", "target": target, "metric": metric, "value": val, "count": len(v)}

    return {"kind": "overview", "metrics": prediction_metrics(df)}


if ask and question.strip():
    try:
        if find_customer(question, full_df) is not None:
            plan = {"type": "customer", "group_by": None, "target_column": None, "metric": "customers", "top_n": 1, "sort": "desc"}
        else:
            try:
                raw_plan = get_plan(question, filtered)
                plan = normalize_plan(raw_plan, question, filtered)
            except Exception:
                plan = fallback_plan(question, filtered)

        result = execute(
                plan,
                full_df if plan.get("type") == "customer" else filtered,
                question,
            )
        st.markdown("### 📊 Analysis Result")

        if result["kind"] == "overview":
            mm = result["metrics"]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Customers", f"{mm['customers']:,}")
            q2.metric("Predicted Churners", f"{mm['predicted_churners']:,}")
            q3.metric("Predicted Churn Rate", f"{mm['predicted_churn_rate']:.1%}")
            q4.metric("Avg Churn Probability", f"{mm['avg_probability']:.1%}")
            st.info(f"There are {mm['predicted_churners']:,} predicted churners out of {mm['customers']:,} customers in the current dashboard view.")

        elif result["kind"] == "group":
            t = result["table"]
            group = result["group"]
            value = result["value"]
            if t.empty:
                st.warning("No segment data is available for this question.")
            else:
                top = t.iloc[0]
                q1, q2, q3 = st.columns(3)
                q1.metric("Customers in View", f"{len(filtered):,}")
                q2.metric("Predicted Churners", f"{int(pd.to_numeric(filtered['Predicted_Churn'], errors='coerce').fillna(0).sum()):,}")
                if value == "Predicted_Churn_Rate":
                    q3.metric("Top Segment Churn Rate", f"{float(top[value]):.1%}")
                else:
                    q3.metric("Top Segment", str(top[group]))
                st.bar_chart(t.set_index(group)[value])
                show = t.copy()
                show["Predicted_Churn_Rate"] = (show["Predicted_Churn_Rate"] * 100).round(2).astype(str) + "%"
                show["Avg_Churn_Probability"] = (show["Avg_Churn_Probability"] * 100).round(2).astype(str) + "%"
                st.dataframe(show, width="stretch", hide_index=True)
                st.success(f"{top[group]} is the highest segment in the requested ranking with {int(top['Predicted_Churners']):,} predicted churners out of {int(top['Customers']):,} customers ({float(top['Predicted_Churn_Rate']):.1%} predicted churn rate).")

        elif result["kind"] == "numeric":
            st.metric(f"{result['metric'].title()} {result['target'].replace('_', ' ')}", f"{result['value']:,.2f}")
            numeric_series = pd.to_numeric(filtered[result["target"]], errors="coerce")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Column": result["target"],
                        "Records": result["count"],
                        "Average": numeric_series.mean(),
                        "Minimum": numeric_series.min(),
                        "Maximum": numeric_series.max(),
                    }
                ]),
                width="stretch",
                hide_index=True,
            )

        elif result["kind"] == "customer":
            row = result["row"]
            if row is None:
                st.warning("Customer was not found in the current filtered data.")
            else:
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Churn Probability", f"{float(row['Churn_Probability_Percent']):.1f}%")
                q2.metric("Risk", str(row["Risk_Level"]))
                q3.metric("Priority", str(row["Retention_Priority"]))
                q4.metric("Predicted Churn", "Yes" if int(row["Predicted_Churn"]) == 1 else "No")
                st.dataframe(
                    pd.DataFrame(
                        [(c.replace("_", " "), str(row[c])) for c in row.index if c != "Customer_ID"],
                        columns=["Column", "Value"],
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.info(str(row.get("Retention_Recommendation", "No recommendation available.")))

        else:
            st.warning("No suitable numeric column was found for this question.")

        # Groq explains only Python-calculated results; it does not calculate dashboard numbers.
        explanation_data = {k: v for k, v in result.items() if k != "row"}
        if result["kind"] == "group":
            explanation_data["rows"] = result["table"].head(10).to_dict(orient="records")
        explanation_prompt = f"""Explain this telecom BI result in simple English. Never change or invent numbers. Give one direct answer and up to 3 short insights. Do not claim causation. Clearly say predicted churn, not confirmed churn.

User question: {question}
Calculated result: {json.dumps(explanation_data, default=str)}"""
        try:
            st.markdown("### 🤖 Groq Business Insights")
            st.markdown(groq_call(explanation_prompt))
        except Exception as exc:
            st.caption(f"Groq explanation unavailable: {exc}")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


# ----------------------------- Retention list -------------------------------
st.divider()
st.subheader("📊 Retention Priority List")
rv = filtered[filtered["Predicted_Churn"].astype(int) == 1].copy()
if rv.empty:
    rv = filtered.copy()
cols = [
    c for c in [
        "Customer_ID",
        "Churn_Probability_Percent",
        "Risk_Level",
        "Retention_Priority",
        "Contract",
        "Tenure_in_Months",
        "Monthly_Charge",
        "Retention_Recommendation",
    ]
    if c in rv.columns
]
st.dataframe(rv[cols].sort_values("Churn_Probability_Percent", ascending=False), width="stretch", hide_index=True)