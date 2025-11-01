# Data Quality Check Candidates

## Credit Card Fraud (`creditcard.csv`)
- Label integrity: `Class` must only contain {0, 1}.
- Missing values: All numeric feature columns should remain complete.
- Transaction amount: `Amount` must be non-negative.
- Feature drift: Monitor each PCA-like feature (`V1`-`V28`) and `Amount` for range breaches versus EDA baselines.
- Class balance: Track the fraud rate to surface ingestion or sampling issues.

## Bank Customer Churn (`Churn_Modelling.csv`)
- Uniqueness: `CustomerId` should be unique and non-null.
- Target domain: `Exited` must only contain {0, 1}.
- Mandatory demographics: `Geography`, `Gender`, `Age`, `CreditScore`, `Balance`, `EstimatedSalary` should not be missing.
- Numeric ranges: Credit score, balance, age, and estimated salary should stay within profiled bounds.
- Categorical domains: Geography and Gender values limited to known categories.
