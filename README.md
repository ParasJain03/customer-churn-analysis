# 📊 AI-Driven Telecom Customer Churn Prediction & Retention Recommendation

🚀 **End-to-End Telecom Analytics | SQL | Power BI | Machine Learning | Retention Intelligence**

An end-to-end telecom customer analytics system that processes customer data using **SQL Server**, analyzes churn in **Power BI**, predicts future churn using **Random Forest**, and generates **personalized retention recommendations** for high-risk customers.

This project is extended for **Use Case 12: Telecom Customer Churn Prediction & Retention Recommendation**.

---

# 🎯 Use Case 12 Objective

Telecom companies need to identify customers who are likely to leave and decide **what retention action should be taken for each high-risk customer**.

This project therefore has two connected stages:

1. **Churn Prediction** — estimate the probability that a customer will churn.
2. **Retention Recommendation** — convert churn risk and customer attributes into an actionable retention strategy.

---

# 🔄 Updated End-to-End Pipeline

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
Retention Recommendation Engine
        ↓
Retention Priority
        ↓
Power BI / GenAI-ready Output
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
      Retention Recommendation
                  ↓
       Retention Priority
                  ↓
    Retention_Recommendations.csv
                  ↓
        Power BI / GenAI
```

The existing SQL ETL creates `vw_ChurnData` for `Stayed/Churned` customers and `vw_JoinData` for `Joined` customers. The new retention workflow uses the same architecture rather than replacing it.

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

The original project already achieved approximately **87–88% accuracy** in the stored notebook evaluation.

---

# 📈 Risk Segmentation

The model probability is converted into an operational risk level:

| Churn Probability | Risk Level |
|---:|---|
| `< 40%` | Low |
| `40–70%` | Medium |
| `>= 70%` | High |

This allows retention teams to prioritize customers rather than treating every predicted churner equally.

---

# 🎯 Retention Recommendation Engine

The new `notebooks/retention_recommendation.py` module adds a rule-based business layer after ML prediction.

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

The new workflow writes:

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

This output can be imported into Power BI and can also be passed to a future GenAI layer to generate natural-language retention strategies.

---

# 📊 Dashboard

The existing Power BI dashboard continues to provide:

* Total Customers
* Churn Rate
* Demographics
* Service Usage
* Geographic churn distribution
* Churn reasons
* Predicted churn customers

For Use Case 12, add a **Retention Recommendation** page using `Retention_Recommendations.csv` with:

* High-risk customer count
* Average churn probability
* Customer ID
* Churn probability
* Risk level
* Retention priority
* Recommended action
* Recommendation distribution

---

# 📂 Project Structure

```text
customer-churn-analysis
│
├── dashboard
│   └── churn_dashboard.pbix
│
├── dashboard_Images
│   ├── churn_analysis.png
│   ├── churn_prediction.png
│   └── churn_reason.png
│
├── data
│   ├── raw
│   │   └── Customer_Data.csv
│   │
│   ├── processed
│   │   └── Prediction_Data.xlsx
│   │
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
├── doc
│   └── project_architecture.md
│
└── README.md
```

---

# 🚀 How to Run

### 1️⃣ Setup SQL Server

Run:

```text
sql/churn_etl.sql
```

This creates the existing ETL pipeline and analytical views.

### 2️⃣ Train and generate retention recommendations

From the repository root:

```bash
python notebooks/retention_recommendation.py
```

The script trains the Random Forest model, calculates churn probabilities, predicts future churn for `vw_JoinData`, assigns risk levels, generates retention recommendations, and saves the result to:

```text
data/predictions/Retention_Recommendations.csv
```

### 3️⃣ Power BI

Open:

```text
dashboard/churn_dashboard.pbix
```

Import the new CSV as an additional source for the Retention Recommendation page.

---

# 🧠 Future GenAI Layer

The retention engine deliberately separates **prediction** from **generation**:

```text
Random Forest
      ↓
Churn Probability
      ↓
Business Rules
      ↓
Retention Recommendation
      ↓
GenAI
      ↓
Personalized Retention Strategy
```

GenAI should explain and personalize the structured recommendation rather than replace the supervised churn model.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| SQL Server | Data storage, ETL and analytical views |
| Power BI | Business analytics and dashboards |
| Python | ML and recommendation pipeline |
| Pandas | Data processing |
| Scikit-Learn | Random Forest churn prediction |
| Joblib | Model persistence |
| Jupyter Notebook | Existing ML development |

---

# 💼 Business Value

The upgraded system helps a telecom company:

* Identify customers at high risk of churn
* Quantify churn probability
* Prioritize retention activity
* Recommend targeted retention actions
* Analyze churn drivers
* Provide a foundation for personalized GenAI retention messaging

---

# 👨‍💻 Author

**Paras Jain**  
B.Tech CSE (Artificial Intelligence) — KIET Group of Institutions

GitHub: https://github.com/ParasJain03
