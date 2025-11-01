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

Track ongoing milestones and environment notes in `docs/progress_update.md`.

## Documentation Index (`docs/`)

- `data_sources.md` – Kaggle dataset references, licences, and download guidance.
- `data_quality_checks.md` – Candidate validation rules to be implemented with Great Expectations.
- `progress_update.md` – Snapshot of completed steps, how to rerun key scripts, and virtual-environment caveats.

Additional documentation (Responsible AI, executive summary, etc.) will be added as the project advances.
