# GenAI Retention Layer

The current project now has a GenAI-ready output contract, but **does not make an external LLM/API call yet**.

## Current pipeline

```text
SQL Server
   ↓
vw_JoinData
   ↓
Random Forest
   ↓
Churn Probability
   ↓
Risk + Priority
   ↓
Rule-based Retention Recommendation
   ↓
Retention_Recommendations.csv
```

## Next integration

Connect an approved LLM provider after the CSV generation step:

```text
Retention_Recommendations.csv
          ↓
Customer record
          ↓
LLM prompt template
          ↓
Personalized retention strategy
          ↓
Email / SMS / call-center script
```

The prompt contract is in `genai/retention_prompt_template.md`.

Keep API keys in environment variables and never commit them to GitHub.
