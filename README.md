# FusionGuard Analytics

FusionGuard Analytics demonstrates an end-to-end workflow for credit card fraud detection and customer churn risk modelling, showcasing modern data science and MLOps practices across the full lifecycle.

## Project Highlights

- Data engineering pipelines using Spark-ready scripts (`src/data`)
- Fraud detection models (XGBoost, autoencoder) and churn prediction models (`src/models`)
- MLOps scaffolding (`mlops`) with experiment tracking and deployment hooks
- Notebooks for exploratory analysis (`notebooks/eda`)
- Documentation (`docs`) covering data sources, responsible AI, and executive summaries

## Roadmap Overview

1. **Project Bootstrap** – Plan repository (`data/raw`, `notebooks/eda`, `src/data`, `src/models`, `mlops`, `docs`); stand up Python env with pyspark, scikit-learn, xgboost, lightgbm, mlflow, fastapi, shap, great_expectations.
2. **Gather Kaggle Data** – Request Kaggle API token; download `creditcardfraud` and `bank-customer-churn` datasets into `data/raw`; document licensing in `docs/data_sources.md`.
3. **EDA & Data Quality** – Build fraud/churn notebooks under `notebooks/eda`; export HTML; list candidate checks for Great Expectations.
4. **Data Engineering** – Use PySpark to clean raw data into `data/processed`; create `feature_store.parquet`; implement bias diagnostics (`data_bias_report.json`); script housed in `src/data/build_feature_store.py`.
5. **Fraud Modelling** – Train XGBoost and PyTorch autoencoder (`src/models/fraud/`); evaluate Precision-Recall/AUCPR; generate SHAP explainability; summarise outcomes in `docs/fraud_model_card.md`.
6. **Churn Modelling** – Train LightGBM/CatBoost churn models (`src/models/churn/train_lightgbm.py`); deliver metrics, feature importances, and business insights.
7. **Generative Module** – FastAPI assistance endpoint (`src/agent/service.py`) with retrieval over project docs, optional llama.cpp integration, and safety-aware prompting (see `docs/agent_service.md`).
8. **MLOps & Automation** – Leverage MLflow, Prefect/Airflow, Great Expectations, and Docker; prepare train/deploy pipelines; target low-cost Cloud Run deployment.
9. **Monitoring & Responsible AI** – Implement data/ performance drift detection (`src/monitoring/`); prototype dashboard (Streamlit/Panel); author `docs/responsible_ai.md`.
10. **Demo & Docs** – Finalise README, architecture visuals, reproduction guide, pitch deck (`slides/`), exec summary, and zero-cost playbook; produce video/GIF walkthrough.

## Documentation Index (`docs/`)

- `data_sources.md` – Kaggle dataset references, licences, and download guidance.
- `data_quality_checks.md` – Candidate validation rules to be implemented with Great Expectations.
- `fraud_model_card.md` – Detailed fraud modelling summary, metrics, and SHAP insights (Step 5 deliverable).
- `churn_model_card.md` – LightGBM churn modelling summary and business insights (Step 6 deliverable).
- `agent_service.md` – FastAPI generative assistant design, deployment notes, and persona guidance (Step 7 deliverable).
- `fraud_serving.md` – Benchmark methodology, latency/cost analysis, and serving recommendations.
- `assets/fraud/` – Generated SHAP visuals and tabular importance export.
