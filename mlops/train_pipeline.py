#!/usr/bin/env python3
"""End-to-end fraud model training orchestration pipeline."""

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

from src.models.fraud.explain_xgboost import generate_shap
from src.models.fraud.train_autoencoder import train_autoencoder
from src.models.fraud.train_xgboost import train_xgboost


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
        "--summary-path",
        type=Path,
        default=Path("artifacts/fraud/pipeline_summary.json"),
        help="File to write consolidated pipeline results.",
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, help="Optional MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment", type=str, default="fusionguard-fraud")
    parser.add_argument("--mlflow-run-name", type=str, default="fraud-train-pipeline")
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
            }
        )
        # Parameters
        mlflow.log_params(
            {
                "autoencoder_threshold": summary["autoencoder"]["threshold"],
                "xgb_threshold": summary["xgboost"]["metrics"]["threshold"],
                "autoencoder_device": summary["autoencoder"]["device"],
                "sample_size_shap": summary["shap"]["sample_size"],
            }
        )
        for label, path in artifact_paths.items():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=label)


def main() -> None:
    args = parse_args()
    summary: Dict[str, object] = {"timestamp": datetime.utcnow().isoformat()}

    if args.refresh_feature_store:
        print("🔄 Refreshing feature store...")
        run_feature_store(args.feature_store)
        summary["feature_store_refreshed"] = True
    else:
        summary["feature_store_refreshed"] = False

    print("🚀 Training XGBoost model...")
    xgb_result = train_xgboost(
        feature_store=args.feature_store,
        output_dir=args.fraud_xgb_dir,
    )
    summary["xgboost"] = xgb_result

    print("🚀 Training Autoencoder...")
    ae_result = train_autoencoder(
        feature_store=args.feature_store,
        output_dir=args.fraud_autoencoder_dir,
        device=args.autoencoder_device,
    )
    summary["autoencoder"] = ae_result

    print("🧮 Generating SHAP explainability for XGBoost...")
    shap_result = generate_shap(
        feature_store=args.feature_store,
        model_path=xgb_result["artifact_paths"]["model"],
        output_dir=args.shap_output,
    )
    summary["shap"] = shap_result

    # Write summary
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"✅ Pipeline summary written to {args.summary_path}")

    artifact_paths = {
        "xgboost": Path(xgb_result["artifact_paths"]["model"]),
        "xgboost_metrics": Path(xgb_result["artifact_paths"]["metrics"]),
        "autoencoder": Path(ae_result["artifact_paths"]["model"]),
        "autoencoder_metrics": Path(ae_result["artifact_paths"]["metrics"]),
        "shap": Path(shap_result["artifact_paths"]["importance_csv"]),
    }

    maybe_log_to_mlflow(
        args.mlflow_tracking_uri,
        args.mlflow_experiment,
        args.mlflow_run_name,
        summary,
        artifact_paths,
    )


if __name__ == "__main__":
    main()
