# 📡 AI-Driven Telecom Customer Churn Prediction & Personalized Retention Recommendation

🚀 **End-to-End Telecom Analytics | SQL | Power BI | Machine Learning | Retention Intelligence | Streamlit | GenAI**

An end-to-end telecom customer analytics and retention decision-support system. The project processes customer data using **SQL Server**, analyzes churn using **Power BI**, predicts future churn using **Random Forest**, converts model probability into risk and retention priority, recommends business actions, and provides an optional **GenAI-powered personalized retention strategy** through a Streamlit application.

This project is extended for **Use Case 12: Telecom Customer Churn Prediction & Retention Recommendation**.

---

# 🎯 Use Case 12 Objective

Telecom companies need to identify customers who are likely to leave and decide **what retention action should be taken for each high-risk customer**.

The solution therefore has four layers:

1. **Churn Prediction** — estimate the probability that a customer will churn.
2. **Risk & Priority** — convert probability into an operational risk level and retention priority.
3. **Retention Recommendation** — map customer signals to actionable retention interventions.
4. **GenAI Personalization** — turn the structured recommendation into a customer-specific strategy/message when an approved LLM endpoint is configured.

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
Optional GenAI Personalized Strategy
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
                       GenAI
```

The existing SQL ETL creates `vw_ChurnData` for `Stayed/Churned` customers and `vw_JoinData` for `Joined` customers. The new workflow reuses that architecture rather than replacing it.

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

### Latest local validation

The Use Case 12 pipeline was successfully executed on the project data with:

```text
Accuracy: 86%
Churn precision: 84%
Churn recall: 63%
Churn F1: 72%
ROC-AUC: 0.905
Predicted churners: 371
```

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

The repository now includes:

```text
app.py
```

The application provides:

* Customer filtering
* Churn probability
* Risk level
* Retention priority
* Customer profile
* Rule-based retention recommendation
* Retention priority list
* Optional GenAI personalized strategy

Run it from the repository root:

```bash
streamlit run app.py
```

The app reads the generated `data/predictions/Retention_Recommendations.csv` file.

---

# 🤖 GenAI Integration

The GenAI layer is implemented in `app.py` using the OpenAI Python client and an OpenAI-compatible chat endpoint.

The LLM **does not predict churn**. The Random Forest model remains responsible for churn prediction. The LLM receives the model output, customer attributes, risk level, priority, and rule-based recommendation and converts them into a concise retention strategy.

```text
Random Forest
      ↓
Churn Probability
      ↓
Business Rules
      ↓
Retention Recommendation
      ↓
LLM
      ↓
Personalized Retention Strategy
```

Configure the optional integration using environment variables:

```text
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=optional_openai_compatible_endpoint
OPENAI_MODEL=your_model
```

Use `.env.example` as the template. **Never commit a real API key to GitHub.**

If no API key is configured, the Streamlit application still works and displays the deterministic retention recommendation.

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
│       ├── Predictions.csv.xlsx
│       └── Retention_Recommendations.csv
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
```

### 4️⃣ Launch the Streamlit application

```bash
streamlit run app.py
```

### 5️⃣ Optional: enable GenAI

Set the environment variables before starting Streamlit. For example in Windows CMD:

```bat
set OPENAI_API_KEY=your_key
set OPENAI_MODEL=your_model
```

For an OpenAI-compatible provider, also set:

```bat
set OPENAI_BASE_URL=your_provider_endpoint
```

Then run:

```bash
streamlit run app.py
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| SQL Server | Data storage, ETL and analytical views |
| Power BI | Business analytics and dashboards |
| Python | ML and application layer |
| Pandas | Data processing |
| Scikit-Learn | Random Forest churn prediction |
| Joblib | Model persistence |
| Streamlit | Interactive retention application |
| OpenAI-compatible API | Optional GenAI personalization |

---

# 💼 Business Value

The upgraded system helps a telecom company:

* Identify customers at high risk of churn
* Quantify churn probability
* Prioritize retention activity
* Recommend targeted retention actions
* Analyze churn drivers
* Generate personalized retention communication

---

# ⚠️ Responsible Use

Churn probability is a model estimate, not certainty. Retention actions should be reviewed against actual customer policy, approved offers, and business constraints before being used with customers. The GenAI layer is instructed not to invent customer facts or commercial offers.

---

# 👨‍💻 Author

**Paras Jain**  
B.Tech CSE (Artificial Intelligence) — KIET Group of Institutions

GitHub: https://github.com/ParasJain03
