# FusionGuard Analytics

FusionGuard Analytics demonstrates an end-to-end workflow for credit card fraud detection and customer churn risk modelling, showcasing modern data science and MLOps practices across the full lifecycle.

## Project Highlights

- Data engineering pipelines using Spark-ready scripts (`src/data`)
- Fraud detection models (XGBoost, autoencoder) and churn prediction models (`src/models`)
- MLOps scaffolding (`mlops`) with experiment tracking and deployment hooks
- Notebooks for exploratory analysis (`notebooks/eda`)
- Documentation (`docs`) covering data sources, responsible AI, and executive summaries

## Architecture Overview

```mermaid
flowchart LR
  raw[Kaggle datasets]
  etl[Spark ETL jobs (src/data)]
  fs[Feature store (data/feature_store)]
  train[Training + tracking (mlops/train scripts, MLflow, SHAP)]
  agent[FastAPI agent + llama.cpp (src/agent)]
  monitor[Monitoring & RAI (src/monitoring, docs)]
  deploy[Deployment (Docker, Cloud Run, Prefect)]

  raw --> etl --> fs --> train
  fs --> agent
  fs --> monitor
  train --> agent
  train --> monitor
  agent --> deploy
  monitor --> deploy
```

## Roadmap Overview

1. **Project Bootstrap** – Plan repository (`data/raw`, `notebooks/eda`, `src/data`, `src/models`, `mlops`, `docs`); stand up Python env with pyspark, scikit-learn, xgboost, lightgbm, mlflow, fastapi, shap, great_expectations.
2. **Gather Kaggle Data** – Request Kaggle API token; download `creditcardfraud` and `bank-customer-churn` datasets into `data/raw`; document licensing in `docs/data_sources.md`.
3. **EDA & Data Quality** – Build fraud/churn notebooks under `notebooks/eda`; export HTML; list candidate checks for Great Expectations.
4. **Data Engineering** – Use PySpark to clean raw data into `data/processed`; create `feature_store.parquet`; implement bias diagnostics (`data_bias_report.json`); script housed in `src/data/build_feature_store.py`.
5. **Fraud Modelling** – Train XGBoost and PyTorch autoencoder (`src/models/fraud/`); evaluate Precision-Recall/AUCPR; generate SHAP explainability; summarise outcomes in `docs/fraud_model_card.md`.
6. **Churn Modelling** – Train LightGBM/CatBoost churn models (`src/models/churn/train_lightgbm.py`); deliver metrics, feature importances, and business insights.
7. **Generative Module** – FastAPI assistance endpoint (`src/agent/service.py`) with retrieval over project docs, optional llama.cpp integration, and safety-aware prompting (see `docs/agent_service.md`).
8. **MLOps & Automation** – MLflow-enabled training pipeline with Great Expectations validation, Prefect flow orchestration, Docker packaging, and Cloud Run deployment helpers (`mlops/train_pipeline.py`, `mlops/deploy_pipeline.py`, `mlops/prefect_flow.py`, see `docs/mlops_deployment.md`).
9. **Monitoring & Responsible AI** – Implement data/ performance drift detection (`src/monitoring/`); prototype dashboard (Streamlit/Panel); author `docs/responsible_ai.md`.
10. **Demo & Docs** – Finalise README, architecture visuals, reproduction guide, pitch deck (`slides/`), exec summary, and zero-cost playbook; produce video/GIF walkthrough.

## Documentation Index (`docs/`)

- `data_sources.md` – Kaggle dataset references, licences, and download guidance.
- `data_quality_checks.md` – Candidate validation rules to be implemented with Great Expectations.
- `fraud_model_card.md` – Detailed fraud modelling summary, metrics, and SHAP insights (Step 5 deliverable).
- `churn_model_card.md` – LightGBM churn modelling summary and business insights (Step 6 deliverable).
- `agent_service.md` – FastAPI generative assistant design, deployment notes, and persona guidance (Step 7 deliverable).
- `mlops_deployment.md` – Experiment tracking, validation, Prefect orchestration, and Cloud Run deployment guidance (Step 8 deliverable).
- `fraud_serving.md` – Benchmark methodology, latency/cost analysis, and serving recommendations.
- `assets/fraud/` – Generated SHAP visuals and tabular importance export.
- `responsible_ai.md` – Responsible AI governance, bias mitigation workflow, and compliance guardrails (Step 9 deliverable).

## CREATE HPC Training Quickstart

- Request an interactive session (interruptible queue starts fastest):  
  `srun --pty -p interruptible_cpu -t 01:00:00 -c 8 --mem=32G bash`
- Initialise dependencies on the compute node:  
  `export JAVA_HOME=/scratch/users/<id>/opt/java/jdk-17.0.11+9`  
  `export PATH=$JAVA_HOME/bin:$PATH`  
  `export SPARK_LOCAL_IP=$(hostname -I | awk '{print $1}')`  
  `export PYSPARK_PYTHON=$(which python)`  
  `source .venv_hpc/bin/activate`
- Run the Step 8 pipeline (logs to MLflow experiment `fusionguard-fraud` and writes artifacts under `artifacts/`):  
  ``python mlops/train_pipeline.py --refresh-feature-store --mlflow-run-name "hpc-train-$(date +%F)" --autoencoder-device cuda``
  (CUDA is requested but the script automatically falls back to CPU if GPUs are unavailable.)
- Sync outputs back to your laptop when finished:  
  `rsync -av --delete create:~/fusionguard-analytics/ ~/Desktop/project/fusionguard-analytics/`

## Drift Monitoring Demo

- Launch the Streamlit dashboard locally:  
  `streamlit run src/monitoring/dashboard.py`
- The app samples the latest feature store, compares reference vs. current segments, and surfaces PSI/KS alerts alongside simulated KPI drift.
- Default threshold settings flag alerts when PSI ≥ 0.2 or KPI deltas exceed ±2pp; tweak values in the sidebar to explore mitigation strategies.

## Reproduction Guide

1. **Clone & Environment**
   - `git clone https://github.com/lee0721/fusionguard-analytics.git`
   - `python3.11 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`

2. **Fetch Datasources**
   - Populate `data/raw/` via Kaggle API (creditcardfraud & bank-customer-churn); see `docs/data_sources.md`.

3. **Feature Engineering**
   - Local quick run: `python src/data/build_feature_store.py --feature-store data/feature_store.parquet`
   - HPC scalable run: follow the “CREATE HPC Training Quickstart” section (Spark cluster ready).

4. **Model Training & Tracking**
   - `python mlops/train_pipeline.py --refresh-feature-store --mlflow-run-name "local-dev"`
   - Inspect MLflow runs at `mlflow ui --backend-store-uri artifacts/mlruns`.

5. **API & Monitoring**
   - Launch agent service: `uvicorn src.agent.service:app --reload`
   - Start drift dashboard: `streamlit run src/monitoring/dashboard.py`

6. **Deployment (Optional)**
   - Build container: `docker build -t fusionguard-agent:latest .`
   - Deploy with helper: `python mlops/deploy_pipeline.py --image gcr.io/<project>/fusionguard-agent:latest --project <gcp-project> --push --deploy --allow-unauthenticated`

## Zero-Cost Strategy

- **Compute**: heavy Spark preprocessing and model retraining execute on King’s CREATE HPC interruptible queues (no direct cloud spend). Local development uses laptop CPU/GPU.
- **Serving**: container targets Cloud Run with `--min-instances 0`, so inference scales to zero outside demo hours.
- **Storage**: feature store and artifacts reside in repo-friendly parquet/CSV; MLflow runs default to the filesystem (`artifacts/mlruns`) rather than paid hosted tracking.
- **Tooling**: all frameworks are OSS (PySpark, MLflow, Streamlit, llama.cpp). Monitoring dashboard runs locally or on HPC Jupyter nodes without SaaS fees.
