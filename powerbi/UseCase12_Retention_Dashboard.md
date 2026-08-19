# Use Case 12 — AI-Driven Churn Retention Dashboard

## Purpose

Extend the existing Power BI churn dashboard with a retention decision layer. The existing project already exposes `vw_ChurnData` for historical churn analytics and `vw_JoinData` for future/Joined customers. The Python retention pipeline produces `Retention_Recommendations.csv` with churn probability, risk level, retention priority, and recommended actions.

## Power BI data sources

Load:

1. Existing SQL Server `vw_ChurnData`
2. Existing SQL Server `vw_JoinData`
3. `data/predictions/Retention_Recommendations.csv`

Recommended relationship:

```text
vw_JoinData[Customer_ID]
        1
        │
        │
        *
Retention_Recommendations[Customer_ID]
```

Use a single Customer dimension if the existing report already has one. Otherwise the Customer_ID relationship above is sufficient for the retention page.

## Retention page layout

### KPI cards

- Predicted High-Risk Customers
- Predicted Churners
- Average Churn Probability
- High-Priority Customers
- Potential Revenue at Risk

### Risk distribution

Donut/column chart:

- Axis: `Risk_Level`
- Values: count of `Customer_ID`

### Retention priority

Column chart:

- Axis: `Retention_Priority`
- Values: count of `Customer_ID`

### Churn probability distribution

Histogram or binned column chart:

- X-axis: `Churn_Probability_Percent`
- Values: count of customers

### Customer retention table

Show:

- Customer_ID
- State
- Contract
- Tenure_in_Months
- Monthly_Charge
- Churn_Probability_Percent
- Risk_Level
- Retention_Priority
- Retention_Recommendation

Sort descending by churn probability.

### Recommended slicers

- State
- Contract
- Internet_Type
- Risk_Level
- Retention_Priority
- Payment_Method

## Suggested DAX measures

```DAX
Predicted Churners =
CALCULATE(
    COUNTROWS('Retention_Recommendations'),
    'Retention_Recommendations'[Predicted_Churn] = 1
)
```

```DAX
High Risk Customers =
CALCULATE(
    COUNTROWS('Retention_Recommendations'),
    'Retention_Recommendations'[Risk_Level] = "High"
)
```

```DAX
High Priority Customers =
CALCULATE(
    COUNTROWS('Retention_Recommendations'),
    'Retention_Recommendations'[Retention_Priority] = "High"
)
```

```DAX
Average Churn Probability =
AVERAGE('Retention_Recommendations'[Churn_Probability])
```

```DAX
Potential Revenue at Risk =
SUMX(
    'Retention_Recommendations',
    'Retention_Recommendations'[Monthly_Charge]
        * 'Retention_Recommendations'[Churn_Probability]
)
```

## Business interpretation

The dashboard should answer three questions:

1. **Who is likely to churn?** — probability and risk level.
2. **Who should we contact first?** — retention priority.
3. **What should we offer?** — personalized retention recommendation.

## GenAI extension

The `Retention_Recommendation` column is intentionally concise so it can be passed to a future GenAI layer. A GenAI service can combine the customer's attributes, churn probability, and recommendation with an approved offer catalog to generate:

- personalized customer message
- call-center talking points
- email/SMS draft
- retention offer explanation
- campaign summary

Do not claim that the current repository calls an LLM unless an API integration is actually added. The current implementation is a deterministic ML + business-rule recommendation layer.
