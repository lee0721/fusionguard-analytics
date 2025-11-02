# FusionGuard Analytics

FusionGuard Analytics demonstrates an end-to-end workflow for credit card fraud detection and customer churn risk modelling, showcasing modern data science and MLOps practices across the full lifecycle.

## Project Highlights

- Data engineering pipelines using Spark-ready scripts (`src/data`)
- Fraud detection models (XGBoost, autoencoder) and churn prediction models (`src/models`)
- MLOps scaffolding (`mlops`) with experiment tracking and deployment hooks
- Notebooks for exploratory analysis (`notebooks/eda`)
- Documentation (`docs`) covering data sources, responsible AI, and executive summaries

## Repository Layout

- `data/raw/` – Kaggle source datasets (credit card fraud, bank churn) to be downloaded locally.
- `data/processed/` – Spark-generated cleaned datasets (output by `build_feature_store.py`).
- `src/data/` – Data engineering scripts; currently contains `build_feature_store.py` for feature store generation.
- `src/models/` – Model training code (fraud detection, churn prediction).
- `mlops/` – Pipelines and automation scaffolding (MLflow, Prefect/Airflow, deployment scripts).
- `notebooks/` – Jupyter notebooks for exploratory data analysis and business insight development.
- `docs/` – Project documentation (see below for details).

## Getting Started

1. **Create the Spark-friendly virtual environment** (the original `.venv` is unstable with PySpark):
   ```bash
   python3 -m venv .venv_spark
   source .venv_spark/bin/activate
   pip install pyspark pandas numpy
   ```
2. **Build processed datasets and feature store**:
   ```bash
   python src/data/build_feature_store.py \
     --raw-dir data/raw \
     --processed-dir data/processed \
     --feature-store data/feature_store.parquet \
     --bias-report data/processed/data_bias_report.json
   ```
   This produces cleaned parquet datasets, a consolidated feature store, and a class-imbalance report.
3. **Explore the EDA outputs**: open `notebooks/eda/reports/churn_eda.html` and `notebooks/eda/reports/fraud_eda.html` in a browser, or launch Jupyter with the `.venv_spark` kernel.

## Current Progress

- **Step 3 – EDA:** Both fraud and churn notebooks executed with inline outputs and exported HTML reports; candidate data quality checks gathered in `docs/data_quality_checks.md`.
- **Step 4 – Data Engineering:** PySpark pipeline (`src/data/build_feature_store.py`) generates processed parquet datasets, a reusable `data/feature_store.parquet`, and bias metrics in `data/processed/data_bias_report.json`.
- **Step 5 – Fraud Modelling:** XGBoost baseline and PyTorch autoencoder trained using `src/models/fraud/`; metrics and inference artifacts live in `artifacts/fraud/`, and SHAP explainability exports plus the written model card are captured in `docs/assets/fraud/` and `docs/fraud_model_card.md`.

### Recent Results (Step 5)
- XGBoost: precision 0.8817, recall 0.8367, F1 0.8586, ROC AUC 0.9768, AUCPR 0.8797.
- Autoencoder: precision 0.0539, recall 0.8571, F1 0.1015, ROC AUC 0.9617, AUCPR 0.6433 (high-recall anomaly filter).
- Key SHAP features: V14, V4, V12, V10, V3, indicating which anonymised PCA components drive credit-card fraud alerts.

## Documentation Index (`docs/`)

- `data_sources.md` – Kaggle dataset references, licences, and download guidance.
- `data_quality_checks.md` – Candidate validation rules to be implemented with Great Expectations.
- `fraud_model_card.md` – Detailed fraud modelling summary, metrics, and SHAP insights (Step 5 deliverable).
- `assets/fraud/` – Generated SHAP charts (`xgboost_shap_importance.png`, `xgboost_shap_summary.png`) and tabular importance export for reporting.

Additional documentation (Responsible AI, executive summary, etc.) will be added as the project advances.
