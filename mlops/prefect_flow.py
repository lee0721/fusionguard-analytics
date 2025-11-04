#!/usr/bin/env python3
"""Prefect orchestration for the FusionGuard training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger, task

from .train_pipeline import (
    DEFAULT_MLFLOW_STORE,
    DEFAULT_VALIDATION_REPORT,
    run_training_pipeline,
)


@task(name="run-training-pipeline", retries=0)
def execute_pipeline_task(
    refresh_feature_store: bool,
    feature_store: Path,
    summary_path: Path,
    validation_report: Path | None,
    skip_churn: bool,
    skip_validation: bool,
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment: str,
    mlflow_run_name: str,
    churn_device: str,
    autoencoder_device: str,
) -> dict:
    logger = get_run_logger()
    logger.info("Starting FusionGuard training pipeline")
    summary = run_training_pipeline(
        refresh_feature_store=refresh_feature_store,
        feature_store=feature_store,
        fraud_xgb_dir=Path("artifacts/fraud/xgboost"),
        fraud_autoencoder_dir=Path("artifacts/fraud/autoencoder"),
        shap_output=Path("docs/assets/fraud"),
        autoencoder_device=autoencoder_device,
        skip_churn=skip_churn,
        churn_lightgbm_dir=Path("artifacts/churn/lightgbm"),
        churn_device=churn_device,
        validate_data=not skip_validation,
        validation_report=validation_report,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
    )
    logger.info("Pipeline completed successfully")
    return summary


@task(name="persist-summary")
def persist_summary(summary: dict, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    get_run_logger().info("Summary written to %s", summary_path)


@flow(name="fusionguard-training-flow")
def fusionguard_training_flow(
    refresh_feature_store: bool = False,
    feature_store: Path = Path("data/feature_store.parquet"),
    summary_path: Path = Path("artifacts/fraud/pipeline_summary.json"),
    validation_report: Path = DEFAULT_VALIDATION_REPORT,
    skip_churn: bool = False,
    skip_validation: bool = False,
    autoencoder_device: str = "cuda",
    churn_device: str = "cpu",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "fusionguard-fraud",
    mlflow_run_name: str = "prefect-training-run",
) -> dict:
    """Prefect flow wrapper around the training pipeline.

    Invoke locally with:

        prefect run python -m mlops.prefect_flow
    """

    if mlflow_tracking_uri is None:
        DEFAULT_MLFLOW_STORE.mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = f"file://{DEFAULT_MLFLOW_STORE.resolve()}"

    validation_report_path = None if skip_validation else validation_report

    summary = execute_pipeline_task(
        refresh_feature_store=refresh_feature_store,
        feature_store=feature_store,
        summary_path=summary_path,
        validation_report=validation_report_path,
        skip_churn=skip_churn,
        skip_validation=skip_validation,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        churn_device=churn_device,
        autoencoder_device=autoencoder_device,
    )

    persist_summary(summary=summary, summary_path=summary_path)
    return summary


if __name__ == "__main__":
    fusionguard_training_flow()
