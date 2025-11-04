# Responsible AI & Risk Governance (Step 9)

This guide summarises how FusionGuard approaches Responsible AI across the model lifecycle, aligning with financial-services and JD expectations.

## 1. Ethical Principles & Use Scope

- **Intended purpose:** Detect card fraud and customer churn to safeguard revenue while avoiding unnecessary friction for legitimate customers.
- **Fairness & inclusivity:** Models must be evaluated for disparate impact across demographic segments where legally available (or proxy variables when direct attributes are unavailable). Mitigation actions include threshold adjustment, alert review workflows, and bias-aware feature engineering.
- **Transparency & explainability:** SHAP explanations (Step 5) and training summaries are surfaced to analysts through the model cards and dashboards. Decisions fed into downstream systems must include reason codes when feasible.

## 2. Data Governance & Bias Management

- **Source vetting:** All datasets (Kaggle fraud/churn) are documented in `docs/data_sources.md` with licensing and provenance notes.
- **Data quality controls:** PySpark pipelines (`src/data/build_feature_store.py`) enforce schema, null checks, and bias diagnostics. Step 8 introduced a validation layer (`mlops/data_validation.py`) with clear failure modes.
- **Bias analyses:** `data/processed/data_bias_report.json` stores parity metrics created during feature-store refresh. Analysts should review this file after each training run and compare to historical baselines.
- **Remediation loop:** If bias metrics breach agreed thresholds, trigger model rollback and update the feature store with de-biased features (e.g., reweighting, optimized thresholds, or post-processing calibration).

## 3. Model Risk Management

- **Performance monitoring:** `src/monitoring/drift_detection.py` computes population stability, KS tests, and KPI deltas. The Streamlit dashboard (`streamlit run src/monitoring/dashboard.py`) demonstrates how alerts surface to analysts.
- **Alert thresholds:** Default PSI ≥ 0.2 or KS *p* ≤ 0.05 raise a data drift alert. KPI changes beyond ±2pp (configurable) generate performance alerts.
- **Incident handling:** When alerts trigger, route to an on-call analyst for triage. Actions include deeper slice analysis, retraining, or temporarily tightening manual review rules.
- **Model inventory:** Log each training run to MLflow (`artifacts/mlruns`) with versioned assets. Maintain a change log summarising hyperparameters, data cuts, and approval status.

## 4. Compliance & Privacy

- **Data minimisation:** Only store features strictly required for the fraud/churn objectives. Remove direct identifiers before modelling; retain `entity_id` solely as a join key inside secure environments.
- **Access controls:** Feature store parquet and MLflow artifacts should live in restricted buckets/directories with role-based access (RBAC) and least-privilege policies.
- **Regulatory alignment:** Adhere to GDPR/UK GDPR for EU cardholders, PCI DSS for card-related data handling, and local banking regulations. Maintain audit trails for model decisions where legally mandated.
- **Third-party models:** Any external LLMs (e.g., llama.cpp helper) must be evaluated for leakage risks and used with strict prompt sanitisation (`docs/agent_service.md`).

## 5. Human Oversight & Governance

- **Approval workflow:** Major releases require sign-off from data science, risk, and compliance stakeholders. Checklists should cover dataset changes, validation results, monitoring dashboards, and Responsible AI sign-off.
- **Documentation:** Keep model cards, data sheets, and monitoring summaries current. Archive previous versions for traceability.
- **Training & awareness:** Provide onboarding material so analysts understand drift alerts, SHAP explanations, and ethical considerations when overruling models.

## 6. Continuous Improvement Plan

1. **Automate monitoring jobs** via Prefect/Airflow to run `compute_drift_report` on daily scoring data.
2. **Integrate alerting** with Slack/Teams or incident management tools when PSI/KS/KPI thresholds breach.
3. **Enhance bias analytics** by incorporating fairness metrics (e.g., equal opportunity, demographic parity) as data access permits.
4. **Champion responsible experimentation** by running A/B or shadow deployments before model rollouts, measuring customer impact and complaint volumes.
5. **Review cadence:** Conduct quarterly Responsible AI reviews covering data changes, drift incidents, remediation actions, and regulatory updates.

By operationalising the safeguards above, FusionGuard maintains trustworthy fraud and churn detection while aligning with Responsible AI expectations.
