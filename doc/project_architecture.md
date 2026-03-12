# 📊 Customer Churn Analysis – Project Architecture

## 📌 Overview

This project implements an **end-to-end data analytics pipeline** to analyze and predict telecom customer churn.

The system integrates:

* **SQL Server** for ETL processing
* **Power BI** for business intelligence dashboards
* **Python Machine Learning models** for churn prediction

The goal is to identify **customers at risk of leaving** and generate **actionable insights for retention strategies**.

---

# 🎯 Project Objectives

* Analyze historical customer churn behavior
* Identify key factors influencing churn
* Build a predictive machine learning model
* Provide actionable insights through interactive dashboards

---

# 🔄 Analytics Workflow

```
Raw Data
   ↓
ETL Pipeline
   ↓
Analytical Database
   ↓
Power BI Dashboard
   ↓
Machine Learning Model
   ↓
Churn Prediction
```

---

# 🏗️ System Architecture

The project architecture follows a structured data pipeline:

```
Raw Dataset (CSV)
        ↓
SQL Server Database
        ↓
Staging Table (stg_Churn)
        ↓
Data Cleaning & Transformation
        ↓
Production Table (prod_Churn)
        ↓
SQL Views (vw_ChurnData, vw_JoinData)
        ↓
Power BI Dashboard
        ↓
Machine Learning Model (Random Forest)
        ↓
Predicted Churn Customers
```

---

# 🖼️ Architecture Visualization

<p align="center">
<img src="../main/dashboard_Images/churn_analysis.png" width="900">
</p>

---

# 📂 Data Source

**Dataset:** Telecom Customer Churn Dataset

**Total Records:** **7,043 Customers**

The dataset includes:

* Customer demographic information
* Service subscription details
* Account and billing information
* Revenue metrics
* Customer churn status
* Churn category and churn reason

This dataset supports both **descriptive analytics** and **predictive machine learning modeling**.

---

# ⚙️ Step 1 – ETL Pipeline (SQL Server)

The first step of the project involves building an **ETL pipeline using Microsoft SQL Server**.

---

## 🗄️ Database Creation

```sql
CREATE DATABASE db_Churn;
```

A dedicated database is created to store and process the churn dataset.

---

## 📥 Data Import

The raw dataset (CSV file) is imported into a staging table using **SQL Server Import Wizard**.

### Staging Table

```
stg_Churn
```

### Purpose of Staging Table

* Store raw imported data
* Perform validation checks
* Prepare data for transformation

---

## 🔍 Data Exploration

Basic SQL queries were used to analyze dataset distribution.

```sql
SELECT Gender, COUNT(Gender) AS TotalCount
FROM stg_Churn
GROUP BY Gender;
```

This analysis helps understand:

* Customer demographics
* Contract distribution
* Customer status
* Geographic distribution

---

## ✅ Data Quality Checks

SQL queries were used to identify missing values.

```sql
SELECT
SUM(CASE WHEN Customer_ID IS NULL THEN 1 ELSE 0 END) AS Customer_ID_Null_Count
FROM stg_Churn;
```

This ensures data inconsistencies are detected before transformation.

---

## 🧹 Data Cleaning & Transformation

Missing values were handled using SQL transformations such as:

```
ISNULL(Value_Deal, 'None')
```

Cleaned data is inserted into the production table:

```
prod_Churn
```

This table represents the **final structured dataset used for analytics**.

---

# 📊 Analytical Views

Two SQL views simplify data access.

## 1️⃣ Historical Churn Data

```
vw_ChurnData
```

Contains customers who:

* Stayed
* Churned

Used for:

* Churn analysis
* Machine learning model training

---

## 2️⃣ New Customer Data

```
vw_JoinData
```

Contains newly joined customers used for **future churn prediction**.

---

# 📊 Step 2 – Power BI Data Transformation

Power BI prepares the dataset for visualization and analysis.

### Calculated Columns

**Churn Status**

```
Churned → 1
Stayed → 0
```

**Monthly Charge Groups**

```
< 20
20 – 50
50 – 100
> 100
```

**Age Groups**

```
< 20
20 – 35
36 – 50
> 50
```

**Tenure Groups**

```
< 6 months
6 – 12 months
12 – 18 months
18 – 24 months
> 24 months
```

These transformations help in **segmenting customers for behavioral analysis**.

---

# 📈 Step 3 – Business Metrics

Key performance indicators (KPIs) were created using **Power BI measures**.

| Metric          | Formula                                             |
| --------------- | --------------------------------------------------- |
| Total Customers | COUNT(Customer_ID)                                  |
| Total Churn     | SUM(Churn Status)                                   |
| Churn Rate      | Total Churn / Total Customers                       |
| New Joiners     | COUNT(Customer_ID WHERE Customer_Status = 'Joined') |

These KPIs provide insights into **customer churn trends and business performance**.

---

# 📊 Step 4 – Power BI Dashboard

The Power BI dashboard provides multiple analytical perspectives.

## Churn Analysis Dashboard

<p align="center">
<img src="../main/dashboard_Images/churn_analysis.png" width="900">
</p>

### Dashboard Insights

* Total Customers
* New Joiners
* Total Churn
* Churn Rate
* Customer demographics
* Geographic churn distribution
* Service usage insights

---

## Churn Reason Analysis

<p align="center">
<img src="../main/dashboard_Images/churn_reason.png" width="700">
</p>

This visualization identifies the primary reasons customers leave, such as:

* Network reliability issues
* High service charges
* Limited service availability
* Poor self-service experience

Understanding these drivers helps businesses design **better retention strategies**.

---

# 🤖 Step 5 – Machine Learning Model

A **Random Forest Classification Model** is used to predict customer churn.

---

## Algorithm Used

**Random Forest Classifier**

### Reasons for Selection

* Handles categorical variables effectively
* Reduces overfitting through ensemble learning
* Provides feature importance insights

---

## Data Preparation

Steps performed in Python:

* Import dataset from SQL export
* Remove unnecessary columns
* Encode categorical variables
* Split dataset into training and testing sets

```
Training Data: 80%
Testing Data: 20%
```

---

## Model Training

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

---

## Model Evaluation

Evaluation metrics used:

* Confusion Matrix
* Precision
* Recall
* F1 Score

**Model Accuracy:** **88%**

---

## Feature Importance

The model identifies key drivers of churn:

* Contract type
* Tenure duration
* Monthly charges
* Internet service usage

Feature importance visualization highlights the **most impactful variables influencing churn**.

---

# 📉 Step 6 – Churn Prediction Dashboard

<p align="center">
<img src="../main/dashboard_Images/churn_prediction.png" width="900">
</p>

The trained model is applied to **newly joined customers**.

Predicted churn results are exported as:

```
Predictions.csv
```

This dataset contains **customers likely to churn in the future**.

---

# 💼 Business Value

This project enables businesses to:

* Identify **high-risk customers**
* Launch **targeted retention campaigns**
* Improve **service offerings**
* Reduce **customer churn rate**

---

# 🛠️ Technology Stack

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| SQL Server       | Data storage & ETL pipeline     |
| Power BI         | Data visualization & dashboards |
| Python           | Machine learning model          |
| Pandas / NumPy   | Data processing                 |
| Scikit-Learn     | Random Forest implementation    |
| Jupyter Notebook | Model development               |

---

# 🚀 Future Improvements

Possible enhancements include:

* Deploying the ML model as a **REST API**
* Automating ETL pipelines
* Real-time churn prediction
* Cloud deployment using **AWS or Azure**

---

# 📌 Conclusion

This project demonstrates a **complete end-to-end data analytics workflow** combining:

* SQL data engineering
* Power BI business intelligence dashboards
* Machine learning models

to solve a **real-world churn prediction problem** and help businesses make **data-driven retention decisions**.
