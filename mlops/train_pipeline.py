#!/usr/bin/env python3
"""End-to-end fraud and churn model training orchestration pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.data_validation import validate_feature_store
from src.models.churn.train_lightgbm import train_lightgbm as train_churn_lightgbm
from src.models.fraud.explain_xgboost import generate_shap
from src.models.fraud.train_autoencoder import train_autoencoder
from src.models.fraud.train_xgboost import train_xgboost


DEFAULT_MLFLOW_STORE = PROJECT_ROOT / "artifacts" / "mlruns"
DEFAULT_VALIDATION_REPORT = PROJECT_ROOT / "artifacts" / "validation" / "feature_store_validation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fraud modelling training pipeline.")
    parser.add_argument(
        "--refresh-feature-store",
        action="store_true",
        help="Rebuild the feature store before training.",
    )
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to consolidated feature store parquet.",
    )
    parser.add_argument(
        "--fraud-xgb-dir",
        type=Path,
        default=Path("artifacts/fraud/xgboost"),
        help="Output directory for XGBoost artifacts.",
    )
    parser.add_argument(
        "--fraud-autoencoder-dir",
        type=Path,
        default=Path("artifacts/fraud/autoencoder"),
        help="Output directory for autoencoder artifacts.",
    )
    parser.add_argument(
        "--shap-output",
        type=Path,
        default=Path("docs/assets/fraud"),
        help="Directory for SHAP visualisations.",
    )
    parser.add_argument(
        "--autoencoder-device",
        type=str,
        default="cuda",
        help="Device for autoencoder training (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--skip-churn",
        action="store_true",
        help="Skip the churn LightGBM training step.",
    )
    parser.add_argument(
        "--churn-lightgbm-dir",
        type=Path,
        default=Path("artifacts/churn/lightgbm"),
        help="Output directory for churn LightGBM artifacts.",
    )
    parser.add_argument(
        "--churn-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device for churn LightGBM training.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Great Expectations validation of the feature store.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
        help="Where to write the Great Expectations validation result (JSON).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("artifacts/fraud/pipeline_summary.json"),
        help="File to write consolidated pipeline results.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        help="Override MLflow tracking URI. Defaults to a local ./artifacts/mlruns store.",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="fusionguard-fraud")
    parser.add_argument("--mlflow-run-name", type=str, default="fraud-train-pipeline")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if a tracking URI is available.",
    )
    return parser.parse_args()


def run_feature_store(feature_store: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "src" / "data" / "build_feature_store.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--feature-store",
        str(feature_store),
    ]
    subprocess.run(cmd, check=True)


def maybe_log_to_mlflow(
    tracking_uri: Optional[str],
    experiment: str,
    run_name: str,
    summary: Dict[str, object],
    artifact_paths: Dict[str, Path],
) -> None:
    if not tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        # Metrics
        mlflow.log_metrics(
            {
                "xgb_precision": summary["xgboost"]["metrics"]["precision"],
                "xgb_recall": summary["xgboost"]["metrics"]["recall"],
                "xgb_aucpr": summary["xgboost"]["metrics"]["average_precision"],
                "ae_precision": summary["autoencoder"]["metrics"]["precision"],
                "ae_recall": summary["autoencoder"]["metrics"]["recall"],
                "ae_aucpr": summary["autoencoder"]["metrics"]["average_precision"],
                **(
                    {
                        "churn_precision": summary["churn"]["metrics"]["precision"],
                        "churn_recall": summary["churn"]["metrics"]["recall"],
                        "churn_aucpr": summary["churn"]["metrics"]["average_precision"],
                    }
                    if "churn" in summary and "metrics" in summary["churn"]
                    else {}
                ),
            }
        )
        # Parameters
        mlflow.log_params(
            {
                "autoencoder_threshold": summary["autoencoder"]["threshold"],
                "xgb_threshold": summary["xgboost"]["metrics"]["threshold"],
                "autoencoder_device": summary["autoencoder"]["device"],
                "sample_size_shap": summary["shap"]["sample_size"],
                **(
                    {
                        "churn_threshold": summary["churn"]["metrics"]["threshold"],
                        "churn_device": summary["churn"]["device"],
                    }
                    if "churn" in summary and "metrics" in summary["churn"]
                    else {}
                ),
            }
        )
        for label, path in artifact_paths.items():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=label)


def run_training_pipeline(
    *,
    refresh_feature_store: bool = False,
    feature_store: Path = Path("data/feature_store.parquet"),
    fraud_xgb_dir: Path = Path("artifacts/fraud/xgboost"),
    fraud_autoencoder_dir: Path = Path("artifacts/fraud/autoencoder"),
    shap_output: Path = Path("docs/assets/fraud"),
    autoencoder_device: str = "cuda",
    skip_churn: bool = False,
    churn_lightgbm_dir: Path = Path("artifacts/churn/lightgbm"),
    churn_device: str = "cpu",
    validate_data: bool = True,
    validation_report: Path | None = DEFAULT_VALIDATION_REPORT,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: str = "fusionguard-fraud",
    mlflow_run_name: str = "fraud-train-pipeline",
) -> Dict[str, object]:
    """Execute the end-to-end training pipeline and return a summary dictionary."""

    summary: Dict[str, object] = {"timestamp": datetime.utcnow().isoformat()}

    if refresh_feature_store:
        print("🔄 Refreshing feature store...")
        run_feature_store(feature_store)
        summary["feature_store_refreshed"] = True
    else:
        summary["feature_store_refreshed"] = False

    validation_info: Dict[str, object]
    if validate_data:
        print("🧪 Validating feature store with Great Expectations...")
        validation_result = validate_feature_store(
            feature_store_path=feature_store,
            report_path=validation_report,
        )
        validation_info = {
            "success": validation_result.get("success", False),
            "statistics": validation_result.get("statistics"),
            "report_path": str(validation_report) if validation_report else None,
        }
    else:
        validation_info = {"skipped": True}
    summary["validation"] = validation_info

    print("🚀 Training XGBoost model...")
    xgb_result = train_xgboost(
        feature_store=feature_store,
        output_dir=fraud_xgb_dir,
    )
    summary["xgboost"] = xgb_result

    print("🚀 Training Autoencoder...")
    ae_result = train_autoencoder(
        feature_store=feature_store,
        output_dir=fraud_autoencoder_dir,
        device=autoencoder_device,
    )
    summary["autoencoder"] = ae_result

    churn_result = None
    if skip_churn:
        print("⚠️ Skipping churn LightGBM training step (per --skip-churn flag).")
        summary["churn"] = {"skipped": True}
    else:
        print("🚀 Training LightGBM churn model...")
        churn_result = train_churn_lightgbm(
            feature_store=feature_store,
            output_dir=churn_lightgbm_dir,
            device=churn_device,
        )
        summary["churn"] = churn_result

    print("🧮 Generating SHAP explainability for XGBoost...")
    shap_result = generate_shap(
        feature_store=feature_store,
        model_path=xgb_result["artifact_paths"]["model"],
        output_dir=shap_output,
    )
    summary["shap"] = shap_result

    artifact_paths = {
        "xgboost": Path(xgb_result["artifact_paths"]["model"]),
        "xgboost_metrics": Path(xgb_result["artifact_paths"]["metrics"]),
        "autoencoder": Path(ae_result["artifact_paths"]["model"]),
        "autoencoder_metrics": Path(ae_result["artifact_paths"]["metrics"]),
        "shap": Path(shap_result["artifact_paths"]["importance_csv"]),
    }
    if churn_result:
        artifact_paths.update(
            {
                "churn_model": Path(churn_result["artifact_paths"]["model"]),
                "churn_metrics": Path(churn_result["artifact_paths"]["metrics"]),
                "churn_predictions": Path(churn_result["artifact_paths"]["predictions"]),
            }
        )

    if validation_report and validation_report.exists():
        artifact_paths["validation_report"] = validation_report

    maybe_log_to_mlflow(
        mlflow_tracking_uri,
        mlflow_experiment,
        mlflow_run_name,
        summary,
        artifact_paths,
    )

    summary["artifact_paths"] = {key: str(path) for key, path in artifact_paths.items()}
    return summary


def main() -> None:
    args = parse_args()

    if args.no_mlflow:
        tracking_uri: Optional[str] = None
    else:
        if args.mlflow_tracking_uri:
            tracking_uri = args.mlflow_tracking_uri
        else:
            DEFAULT_MLFLOW_STORE.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"file://{DEFAULT_MLFLOW_STORE.resolve()}"

    validation_report = None if args.skip_validation else args.validation_report

    summary = run_training_pipeline(
        refresh_feature_store=args.refresh_feature_store,
        feature_store=args.feature_store,
        fraud_xgb_dir=args.fraud_xgb_dir,
        fraud_autoencoder_dir=args.fraud_autoencoder_dir,
        shap_output=args.shap_output,
        autoencoder_device=args.autoencoder_device,
        skip_churn=args.skip_churn,
        churn_lightgbm_dir=args.churn_lightgbm_dir,
        churn_device=args.churn_device,
        validate_data=not args.skip_validation,
        validation_report=validation_report,
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )

    # Write summary
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"✅ Pipeline summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
