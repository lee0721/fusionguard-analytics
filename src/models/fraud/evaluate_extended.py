#!/usr/bin/env python3
"""Extended evaluation utilities for fraud models (calibration, threshold sweeps)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended evaluation for fraud detection models.")
    parser.add_argument(
        "--xgb-predictions",
        type=Path,
        default=Path("artifacts/fraud/xgboost/validation_predictions.csv"),
        help="CSV with columns [label, probability, prediction].",
    )
    parser.add_argument(
        "--autoencoder-scores",
        type=Path,
        default=Path("artifacts/fraud/autoencoder/validation_scores.csv"),
        help="CSV with columns [label, reconstruction_error].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fraud/evaluation"),
        help="Directory for extended evaluation artifacts.",
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins for calibration curves.")
    parser.add_argument("--threshold-steps", type=int, default=15, help="Number of thresholds to sweep.")
    return parser.parse_args()


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, ThresholdMetrics]:
    rows = []
    best = None
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        ap = average_precision_score(y_true, scores)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": ap,
            }
        )
        if best is None or f1 > best.f1:
            best = ThresholdMetrics(threshold=threshold, precision=precision, recall=recall, f1=f1, average_precision=ap)
    return pd.DataFrame(rows), best


def plot_calibration(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int,
    output_path: Path,
    title: str,
) -> Dict[str, list]:
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return {"predicted": prob_pred.tolist(), "true": prob_true.tolist()}


def evaluate_xgboost(predictions_path: Path, output_dir: Path, n_bins: int, threshold_steps: int) -> Dict[str, object]:
    df = pd.read_csv(predictions_path)
    y = df["label"].to_numpy()
    probs = df["probability"].to_numpy()

    thresholds = np.linspace(0.05, 0.95, threshold_steps)
    sweep_df, best = sweep_thresholds(y, probs, thresholds)
    sweep_path = output_dir / "xgboost_threshold_metrics.csv"
    sweep_df.to_csv(sweep_path, index=False)

    calibration_path = output_dir / "xgboost_calibration.png"
    calibration_data = plot_calibration(y, probs, n_bins, calibration_path, "XGBoost Calibration")

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y, probs)
    pr_path = output_dir / "xgboost_precision_recall.csv"
    pd.DataFrame(
        {
            "precision": pr_precision,
            "recall": pr_recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    ).to_csv(pr_path, index=False)

    return {
        "roc_auc": roc_auc_score(y, probs),
        "average_precision": average_precision_score(y, probs),
        "best_threshold": best.__dict__,
        "threshold_metrics_path": sweep_path,
        "calibration_plot": calibration_path,
        "calibration_data": calibration_data,
        "precision_recall_path": pr_path,
    }


def evaluate_autoencoder(scores_path: Path, output_dir: Path, n_bins: int, threshold_steps: int) -> Dict[str, object]:
    df = pd.read_csv(scores_path)
    y = df["label"].to_numpy()
    errors = df["reconstruction_error"].to_numpy()

    # Normalise errors to [0, 1] for calibration visualisation.
    min_err, max_err = errors.min(), errors.max()
    norm_scores = (errors - min_err) / (max_err - min_err + 1e-12)

    # Use percentiles for threshold sweep (converted back to raw error space).
    percentiles = np.linspace(60, 99.9, threshold_steps)
    thresholds_raw = np.percentile(errors, percentiles)
    sweep_df, best = sweep_thresholds(y, errors, thresholds_raw)
    sweep_df["percentile"] = percentiles
    sweep_path = output_dir / "autoencoder_threshold_metrics.csv"
    sweep_df.to_csv(sweep_path, index=False)

    calibration_path = output_dir / "autoencoder_calibration.png"
    calibration_data = plot_calibration(y, norm_scores, n_bins, calibration_path, "Autoencoder Calibration (error-scaled)")

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y, errors)
    pr_path = output_dir / "autoencoder_precision_recall.csv"
    pd.DataFrame(
        {
            "precision": pr_precision,
            "recall": pr_recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    ).to_csv(pr_path, index=False)

    return {
        "roc_auc": roc_auc_score(y, errors),
        "average_precision": average_precision_score(y, errors),
        "best_threshold": best.__dict__,
        "threshold_metrics_path": sweep_path,
        "calibration_plot": calibration_path,
        "calibration_data": calibration_data,
        "precision_recall_path": pr_path,
        "normalisation": {"min_error": float(min_err), "max_error": float(max_err)},
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "xgboost": evaluate_xgboost(args.xgb_predictions, output_dir, args.n_bins, args.threshold_steps),
        "autoencoder": evaluate_autoencoder(args.autoencoder_scores, output_dir, args.n_bins, args.threshold_steps),
    }

    summary_path = output_dir / "evaluation_extended.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"✅ Extended evaluation saved to {summary_path}")


if __name__ == "__main__":
    main()
