# Executive Summary

**FusionGuard Analytics** delivers a dual-model risk platform that reduces card fraud losses while preserving high-value customers through proactive churn mitigation.

## Business Outcomes

- **Fraud Prevention**: XGBoost + Autoencoder ensemble achieves AUCPR 0.88 on imbalanced Kaggle data, enabling rule engine replacement with explainable SHAP alerts.
- **Retention Lift**: LightGBM churn model surfaces top-decile customers at 2.4× baseline churn rate, guiding targeted incentives.
- **Operational Efficiency**: Prefect orchestrated pipelines rebuild the feature store nightly in <30 minutes on CREATE HPC and log metrics to MLflow for auditability.
- **Responsible AI**: Bias diagnostics, Responsible AI playbook, and drift dashboard ensure regulatory alignment (GDPR/PCI) and rapid remediation.

## Technical Approach

1. **Unified Feature Store**  
   PySpark ETL consolidates fraud + churn datasets; bias report and validation checks guard data quality.

2. **Model Training**  
   - Fraud: XGBoost classifier + PyTorch autoencoder; SHAP explainability and anomaly threshold export.  
   - Churn: LightGBM gradient boosting with class balancing and business-friendly feature importances.

3. **MLOps Automation**  
   Pipeline (`mlops/train_pipeline.py`) orchestrates feature refresh, training, evaluation, SHAP generation, and MLflow logging. Docker + Cloud Run scripts support zero-cost deployment.

4. **Monitoring & Governance**  
   `src/monitoring/drift_detection.py` computes PSI + KS drift stats; Streamlit dashboard visualises alerts. `docs/responsible_ai.md` codifies bias mitigation and escalation workflow.

## Next Steps

- Launch Cloud Run inference endpoint backed by the FastAPI agent, enabling analysts to query models via chat interface.
- Integrate Prefect deployment for automated retraining schedule and Slack alerting when drift thresholds breach.
- Extend fairness analysis with demographic parity metrics once real-world segments are available.
