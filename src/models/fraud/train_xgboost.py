#!/usr/bin/env python3
"""Train an XGBoost fraud detection model and log evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .data_utils import load_fraud_dataset, stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost fraud classifier.")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to the consolidated feature store parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fraud/xgboost"),
        help="Directory to store model artifacts and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max depth.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Boosting rounds.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsample rate.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample rate.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel threads for XGBoost.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification.")
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, entity_ids = load_fraud_dataset(args.feature_store)
    entity_series = entity_ids.rename("entity_id")

    X_train, X_valid, y_train, y_valid = stratified_split(
        features, labels, test_size=args.test_size, random_state=args.random_state
    )

    entity_valid = entity_series.iloc[X_valid.index.to_numpy()]
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    y_valid_prob = clf.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid.to_numpy(), y_valid_prob, threshold=args.threshold)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    prediction_df = pd.DataFrame(
        {
            "entity_id": entity_valid.values,
            "label": y_valid.to_numpy(),
            "probability": y_valid_prob,
            "prediction": (y_valid_prob >= args.threshold).astype(int),
        }
    )
    prediction_df.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    model_path = args.output_dir / "xgboost_model.json"
    clf.get_booster().save_model(model_path)

    feature_importance = pd.Series(
        clf.feature_importances_, index=X_train.columns, name="importance"
    ).sort_values(ascending=False)
    feature_importance.to_csv(args.output_dir / "feature_importance.csv")

    print(f"✅ Metrics saved to {metrics_path}")
    print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    main()
