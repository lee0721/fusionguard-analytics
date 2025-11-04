# MLOps & Deployment (Step 8)

This guide summarises the automation and deployment assets delivered for Step&nbsp;8 of the FusionGuard roadmap.

## Experiment Tracking (MLflow)

The training pipeline (`mlops/train_pipeline.py`) now logs metrics, parameters, and generated artifacts to MLflow. By default, a local file store at `artifacts/mlruns` is used.

```bash
# Run the full pipeline, refreshing the feature store and logging to MLflow
python mlops/train_pipeline.py \
  --refresh-feature-store \
  --mlflow-run-name "daily-training"

# Specify a remote tracking server if desired
python mlops/train_pipeline.py \
  --mlflow-tracking-uri http://mlflow.internal:5000
```

## Data Validation (Lightweight Expectations)

The pipeline validates `data/feature_store.parquet` with a pandas-based routine that mirrors the original Great Expectations checks. When the optional Great Expectations dependency is present, the integration is used transparently; otherwise the fallback keeps the same assertions and report structure without extra requirements. Validation outputs are written to `artifacts/validation/feature_store_validation.json`, and the bias scan remains in `data/processed/data_bias_report.json`. To skip validation entirely:

```bash
python mlops/train_pipeline.py --skip-validation
```

## Prefect Orchestration

The Prefect flow (`mlops/prefect_flow.py`) wraps the training pipeline so it can be scheduled on CREATE or a local agent.

```bash
# Run ad hoc
prefect run python -m mlops.prefect_flow

# Build a deployment for the default agent
prefect deployment build mlops/prefect_flow.py:fusionguard_training_flow \
  -n nightly-train \
  --schedule "0 2 * * *" \
  --apply
```

## Container Image

A Dockerfile is included at the repository root. It packages the FastAPI agent with all dependencies.

```bash
docker build -t ghcr.io/<org>/fusionguard-agent:latest .
```

## Cloud Run Deployment

Use `mlops/deploy_pipeline.py` to orchestrate Docker build, push, and Cloud Run deployment. The script supports `--dry-run` to preview commands.

```bash
python mlops/deploy_pipeline.py \
  --image gcr.io/<project>/fusionguard-agent:latest \
  --project <gcp-project> \
  --push \
  --deploy \
  --allow-unauthenticated \
  --region us-central1
```

Consider setting `--min-instances 0` in Cloud Run to leverage the free tier's automatic scale-to-zero behaviour.

## Running on CREATE HPC

- Launch an interactive session (interruptible queue shortens wait time):  
  `srun --pty -p interruptible_cpu -t 01:00:00 -c 8 --mem=32G bash`
- Configure Spark prerequisites and activate the project virtualenv:
  ```bash
  export JAVA_HOME=/scratch/users/<id>/opt/java/jdk-17.0.11+9
  export PATH=$JAVA_HOME/bin:$PATH
  export SPARK_LOCAL_IP=$(hostname -I | awk '{print $1}')
  export PYSPARK_PYTHON=$(which python)
  source ~/fusionguard-analytics/.venv_hpc/bin/activate
  ```
- Execute the Step 8 pipeline (requests CUDA; the autoencoder gracefully falls back to CPU when GPUs are unavailable):
  ```bash
  python mlops/train_pipeline.py \
    --refresh-feature-store \
    --mlflow-run-name "hpc-train-$(date +%F)" \
    --autoencoder-device cuda
  ```
- Outputs include refreshed parquet datasets under `data/`, validation reports in `artifacts/validation/`, model artifacts in `artifacts/fraud/` and `artifacts/churn/`, and MLflow runs stored under `artifacts/mlruns` (experiment name `fusionguard-fraud`).
- After the run completes, synchronise results back to your workstation (run from your laptop):  
  `rsync -av --delete create:~/fusionguard-analytics/ ~/Desktop/project/fusionguard-analytics/`
