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
6. **Churn Modelling** – Train LightGBM/CatBoost churn models; deliver metrics, feature importances, and business insights.
7. **Generative Module** – Build FastAPI endpoint using open LLM (llama.cpp/Mistral) for explanations and customer support; incorporate prompt safety, optional GPU fine-tune or RAG (faiss) for zero-cost.
8. **MLOps & Automation** – Leverage MLflow, Prefect/Airflow, Great Expectations, and Docker; prepare train/deploy pipelines; target low-cost Cloud Run deployment.
9. **Monitoring & Responsible AI** – Implement data/ performance drift detection (`src/monitoring/`); prototype dashboard (Streamlit/Panel); author `docs/responsible_ai.md`.
10. **Demo & Docs** – Finalise README, architecture visuals, reproduction guide, pitch deck (`slides/`), exec summary, and zero-cost playbook; produce video/GIF walkthrough.
