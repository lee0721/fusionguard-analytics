#!/usr/bin/env python3
"""Generate SHAP explanations for the trained XGBoost fraud model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .data_utils import load_fraud_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SHAP values for fraud XGBoost model.")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to the consolidated feature store parquet.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/fraud/xgboost/xgboost_model.json"),
        help="Path to the trained XGBoost model JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/fraud"),
        help="Directory to store SHAP visualisations.",
    )
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample size for SHAP computation.")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling random state.")
    return parser.parse_args()


def generate_shap(
    feature_store: Path,
    model_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict:
    """Generate SHAP plots and importance CSV for the trained XGBoost model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, _, _ = load_fraud_dataset(feature_store)
    if sample_size and sample_size < len(features):
        sample = features.sample(n=sample_size, random_state=random_state)
    else:
        sample = features

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmatrix = xgb.DMatrix(sample)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmatrix)

    shap.summary_plot(
        shap_values,
        sample,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    bar_path = output_dir / "xgboost_shap_importance.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    shap.summary_plot(
        shap_values,
        sample,
        show=False,
        max_display=20,
    )
    summary_path = output_dir / "xgboost_shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_df = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_path = output_dir / "xgboost_shap_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    return {
        "sample_size": len(sample),
        "artifact_paths": {
            "bar_plot": bar_path,
            "summary_plot": summary_path,
            "importance_csv": importance_path,
        },
        "top_features": importance_df.head(20).to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    result = generate_shap(
        feature_store=args.feature_store,
        model_path=args.model_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    paths = result["artifact_paths"]
    print(f"✅ SHAP bar plot saved to {paths['bar_plot']}")
    print(f"✅ SHAP summary plot saved to {paths['summary_plot']}")


if __name__ == "__main__":
    main()
