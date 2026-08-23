# 📡 AI-Driven Telecom Customer Churn Prediction & Personalized Retention Recommendation

🚀 **End-to-End Telecom Analytics | SQL | Power BI | Machine Learning | Retention Intelligence | Streamlit | GenAI**

An end-to-end telecom customer analytics and retention decision-support system. The project processes customer data using **SQL Server**, analyzes churn using **Power BI**, predicts future churn using **Random Forest**, converts model probability into risk and retention priority, recommends business actions, and provides a **Groq-powered GenAI personalized retention strategy and natural-language AI Data Analyst** through a Streamlit application.

This project is extended for **Use Case 12: Telecom Customer Churn Prediction & Retention Recommendation**.

---

# 🎯 Use Case 12 Objective

Telecom companies need to identify customers who are likely to leave and decide **what retention action should be taken for each high-risk customer**.

The solution therefore has four layers:

1. **Churn Prediction** — estimate the probability that a customer will churn.
2. **Risk & Priority** — convert probability into an operational risk level and retention priority.
3. **Retention Recommendation** — map customer signals to actionable retention interventions.
4. **GenAI Personalization & AI Data Analyst** — use Groq to explain verified analytics in natural language and generate personalized retention strategies.

---

# 🔄 Complete End-to-End Pipeline

```text
Raw Telecom Dataset
        ↓
SQL Server ETL
        ↓
Cleaned Production Dataset
        ↓
SQL Views
        ↓
Power BI Churn Analytics
        ↓
Random Forest Churn Model
        ↓
Churn Probability
        ↓
Risk Segmentation
        ↓
Retention Priority
        ↓
Rule-Based Retention Recommendation
        ↓
Retention_Recommendations.csv
        ↓
Streamlit Customer Retention App
        ↓
Python Data Analytics
        ↓
Groq GenAI
        ↓
Natural-Language Insights / Personalized Strategy
```

---

# 🏗️ System Architecture

```text
Raw Dataset (CSV)
      ↓
SQL Server Database
      ↓
Staging Table: stg_Churn
      ↓
Data Cleaning & Transformation
      ↓
Production Table: prod_Churn
      ↓
Analytical Views
  ┌───────────────┬───────────────┐
  ↓                               ↓
vw_ChurnData                  vw_JoinData
  ↓                               ↓
Historical Churn              Current/Joined Customers
  ↓                               ↓
Random Forest Training        Future Churn Prediction
  └───────────────┬───────────────┘
                  ↓
          Churn Probability
                  ↓
            Risk Level
       Low / Medium / High
                  ↓
       Retention Priority
                  ↓
      Retention Recommendation
                  ↓
    Retention_Recommendations.csv
            ┌─────┴─────┐
            ↓           ↓
        Power BI    Streamlit
                        ↓
                Python Analytics
                        ↓
                     Groq GenAI
```

The existing SQL ETL creates `vw_ChurnData` for `Stayed/Churned` customers and `vw_JoinData` for `Joined` customers. The Use Case 12 workflow reuses that architecture rather than replacing it.

---

# 🤖 Machine Learning

The existing **Random Forest Classifier** is retained.

### Model pipeline

1. Load `vw_ChurnData`
2. Remove customer ID and post-churn fields from model inputs
3. Encode categorical variables
4. Map `Stayed → 0` and `Churned → 1`
5. Train/test split
6. Train Random Forest
7. Evaluate classification performance
8. Generate **churn probability** using `predict_proba()`
9. Generate leakage-safe predictions for the complete cleaned customer population

### Latest local validation

The Use Case 12 pipeline was successfully executed on the project data with:

```text
Accuracy: 86%
Churn precision: 84%
Churn recall: 63%
Churn F1: 72%
ROC-AUC: 0.905
```

The prediction pipeline generates predictions for **6,007 complete cleaned customer records** for dashboard and retention analytics.

These metrics are from the local execution of `notebooks/retention_recommendation.py` and should be regenerated when the data/model changes.

---

# 📈 Risk Segmentation

| Churn Probability | Risk Level |
|---:|---|
| `< 40%` | Low |
| `40–70%` | Medium |
| `>= 70%` | High |

Risk is an operational category; a high-risk customer is **not guaranteed** to churn.

---

# 🎯 Retention Recommendation Engine

`notebooks/retention_recommendation.py` adds a deterministic business-rule layer after ML prediction.

Examples:

| Customer Signal | Recommended Action |
|---|---|
| Month-to-month contract + elevated risk | Incentivize long-term contract |
| High monthly charge + elevated risk | Personalized pricing/loyalty discount |
| Short tenure + elevated risk | Early-tenure loyalty benefit |
| No premium support + elevated risk | Priority/premium support offer |
| Fiber customer + elevated risk | Service review/upgrade consultation |
| No value deal + elevated risk | Evaluate value-plan eligibility |

The engine produces both a **retention recommendation** and a **retention priority**.

---

# 📄 Prediction Output

The workflow writes:

```text
data/predictions/Retention_Recommendations.csv
data/predictions/All_Customer_Predictions.csv
```

Important fields include:

```text
Customer_ID
Churn_Probability
Churn_Probability_Percent
Predicted_Churn
Risk_Level
Retention_Priority
Retention_Recommendation
```

---

# 🖥️ Streamlit Application

The repository includes:

```text
app.py
```

The application provides:

* Customer filtering
* Churn probability
* Risk level
* Retention priority
* Customer profile / Customer 360
* Rule-based retention recommendation
* Retention priority list
* AI Data Analyst for natural-language analytics
* Groq-powered personalized retention strategy

Run it from the repository root:

```bash
streamlit run app.py
```

The app reads the generated prediction CSV files from `data/predictions/`.

---

# 🤖 Groq GenAI Integration

The GenAI layer is implemented in `app.py` using the **Groq Python SDK**.

### API and model

```text
Provider: Groq API
Model: openai/gpt-oss-20b
```

The application uses:

```python
GROQ_MODEL = "openai/gpt-oss-20b"
```

The model is used for the **natural-language explanation and personalized retention strategy layer**. It does **not** replace the Random Forest churn model.

### Architecture

```text
Random Forest
      ↓
Churn Probability
      ↓
Python Analytics / Business Rules
      ↓
Verified Result / Retention Recommendation
      ↓
Groq API
      ↓
openai/gpt-oss-20b
      ↓
Natural-Language Explanation / Personalized Strategy
```

For the **AI Data Analyst**, Python remains the source of truth for numerical calculations. The dashboard data is filtered and aggregated using Python, and the verified result is supplied to Groq for explanation. This reduces the risk of the LLM inventing or incorrectly calculating business metrics.

### Configuration

Set the Groq API key using an environment variable or Streamlit Secrets.

Local `.env` example:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

For Streamlit Cloud, add the same values under **App Settings → Secrets**.

**Never commit a real API key to GitHub.**

If no Groq API key is configured, the core Streamlit dashboard and deterministic retention recommendation layer can still be used; only the GenAI-powered explanation/strategy functionality requires the API.

---

# 💬 AI Data Analyst

The application includes a Power BI-style natural-language analytics layer.

Users can ask questions about:

* Customer counts
* Predicted churners
* Predicted churn rate
* Churn probability
* Gender, contract, state and internet-type segments
* Risk and retention priority
* Individual customer information using Customer ID
* Other supported dashboard analytics

### Accuracy-first design

```text
User Question
      ↓
Python Analytics
      ↓
Filter / Aggregate / Calculate
      ↓
Verified Numerical Result
      ↓
Groq GenAI
      ↓
Simple Business Explanation
```

The key design principle is:

> **Python calculates; GenAI explains.**

This keeps core numerical business metrics deterministic while still giving users a conversational analytics experience.

---

# 📊 Power BI Dashboard

The existing Power BI dashboard continues to provide:

* Total Customers
* Churn Rate
* Demographics
* Service Usage
* Geographic churn distribution
* Churn reasons
* Predicted churn customers

For Use Case 12, `powerbi/UseCase12_Retention_Dashboard.md` documents the additional Retention Intelligence page using `Retention_Recommendations.csv`.

Recommended visuals:

* Predicted churners
* High-risk customers
* High-priority customers
* Average churn probability
* Risk distribution
* Retention recommendation distribution
* Customer retention priority table

---

# 📂 Project Structure

```text
customer-churn-analysis
│
├── app.py
├── requirements.txt
├── .env.example
│
├── dashboard
│   └── churn_dashboard.pbix
│
├── dashboard_Images
│
├── data
│   ├── raw
│   │   └── Customer_Data.csv
│   ├── processed
│   │   └── Prediction_Data.xlsx
│   └── predictions
│       ├── Retention_Recommendations.csv
│       └── All_Customer_Predictions.csv
│
├── notebooks
│   ├── churn_prediction.ipynb
│   └── retention_recommendation.py
│
├── sql
│   └── churn_etl.sql
│
├── powerbi
│   └── UseCase12_Retention_Dashboard.md
│
├── genai
│   ├── README.md
│   └── retention_prompt_template.md
│
├── doc
│
└── README.md
```

---

# 🚀 How to Run

### 1️⃣ Create/activate a Python environment

```bash
python -m venv venv
```

Windows CMD:

```bat
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Generate churn predictions and retention recommendations

```bash
python notebooks\retention_recommendation.py
```

This creates:

```text
data\predictions\Retention_Recommendations.csv
data\predictions\All_Customer_Predictions.csv
```

### 4️⃣ Configure Groq

Create a `.env` file locally or configure Streamlit Secrets:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

### 5️⃣ Launch the Streamlit application

```bash
streamlit run app.py
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| SQL Server | Data storage, ETL and analytical views |
| Power BI | Business analytics and dashboards |
| Python | ML, analytics and application layer |
| Pandas | Data processing and deterministic analytics |
| Scikit-Learn | Random Forest churn prediction |
| Joblib | Model persistence |
| Streamlit | Interactive retention application |
| Groq API | GenAI inference |
| `openai/gpt-oss-20b` | Natural-language analytics explanation and retention strategy |

---

# 💼 Business Value

The upgraded system helps a telecom company:

* Identify customers at high risk of churn
* Quantify churn probability
* Prioritize retention activity
* Recommend targeted retention actions
* Analyze churn drivers
* Ask natural-language questions about prediction data
* Generate personalized retention communication

---

# ⚠️ Responsible Use

Churn probability is a model estimate, not certainty. Retention actions should be reviewed against actual customer policy, approved offers, and business constraints before being used with customers. The GenAI layer is instructed to use verified customer information and not invent customer facts or commercial offers.

---

# 👨‍💻 Author

**Paras Jain**  
B.Tech CSE (Artificial Intelligence) — KIET Group of Institutions

GitHub: https://github.com/ParasJain03
