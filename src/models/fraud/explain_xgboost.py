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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, _, _ = load_fraud_dataset(args.feature_store)
    if args.sample_size and args.sample_size < len(features):
        sample = features.sample(n=args.sample_size, random_state=args.random_state)
    else:
        sample = features

    booster = xgb.Booster()
    booster.load_model(str(args.model_path))
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
    bar_path = args.output_dir / "xgboost_shap_importance.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    shap.summary_plot(
        shap_values,
        sample,
        show=False,
        max_display=20,
    )
    summary_path = args.output_dir / "xgboost_shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_df = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(args.output_dir / "xgboost_shap_importance.csv", index=False)

    print(f"✅ SHAP bar plot saved to {bar_path}")
    print(f"✅ SHAP summary plot saved to {summary_path}")


if __name__ == "__main__":
    main()
