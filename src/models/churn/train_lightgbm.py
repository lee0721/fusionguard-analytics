#!/usr/bin/env python3
"""Train a LightGBM model for customer churn prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_utils import load_churn_dataset, make_class_weights, stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM churn classifier.")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to the consolidated feature store parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/churn/lightgbm"),
        help="Directory to store churn model artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--num-leaves", type=int, default=64, help="Number of leaves.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=-1, help="Tree max depth (-1 for no limit).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="LightGBM device type.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class.")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    metrics["precision_recall_curve"] = {
        "precision": pr_precision.tolist(),
        "recall": pr_recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }
    metrics["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return metrics


def train_lightgbm(
    feature_store: Path,
    output_dir: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.05,
    num_leaves: int = 64,
    n_estimators: int = 600,
    max_depth: int = -1,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Train LightGBM churn classifier and persist artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, entity_ids = load_churn_dataset(feature_store)
    entity_series = entity_ids.rename("entity_id")

    X_train, X_valid, y_train, y_valid = stratified_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    class_weights = make_class_weights(y_train)
    clf = LGBMClassifier(
        objective="binary",
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weights,
        random_state=random_state,
        n_jobs=-1,
        device=device,
        verbose=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")

    y_prob = clf.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid.to_numpy(), y_prob, threshold=threshold)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    predictions_df = pd.DataFrame(
        {
            "entity_id": entity_series.loc[X_valid.index].values,
            "label": y_valid.to_numpy(),
            "probability": y_prob,
            "prediction": (y_prob >= threshold).astype(int),
        }
    )
    predictions_path = output_dir / "validation_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    importance = pd.Series(clf.feature_importances_, index=X_train.columns, name="importance").sort_values(
        ascending=False
    )
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path)

    model_path = output_dir / "lightgbm_model.txt"
    clf.booster_.save_model(model_path)

    return {
        "metrics": metrics,
        "artifact_paths": {
            "metrics": metrics_path,
            "predictions": predictions_path,
            "feature_importance": importance_path,
            "model": model_path,
        },
        "class_weights": class_weights,
        "device": device,
    }


def main() -> None:
    args = parse_args()
    result = train_lightgbm(
        feature_store=args.feature_store,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        device=args.device,
        threshold=args.threshold,
    )
    paths = result["artifact_paths"]
    print(f"✅ Metrics saved to {paths['metrics']}")
    print(f"✅ Model saved to {paths['model']}")


if __name__ == "__main__":
    main()
