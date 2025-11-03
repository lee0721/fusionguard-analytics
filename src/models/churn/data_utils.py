#!/usr/bin/env python3
"""Utilities for loading and splitting churn feature data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_churn_dataset(
    feature_store_path: Path | str,
    dataset_name: str = "churn",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load churn features and labels from the consolidated feature store parquet."""
    feature_store_path = Path(feature_store_path)
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store not found at {feature_store_path}")

    df = pd.read_parquet(feature_store_path)
    filtered = df[df["dataset"] == dataset_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for dataset='{dataset_name}' in feature store.")

    features = pd.json_normalize(filtered["features"].map(json.loads))
    features.columns = [col.replace(".", "_") for col in features.columns]
    # Ensure categorical columns use pandas Categorical dtype so LightGBM can handle them natively
    object_cols = features.select_dtypes(include=["object"]).columns
    for col in object_cols:
        features[col] = features[col].astype("category")
    labels = filtered["label"].astype(int)
    entity_ids = filtered["entity_id"].astype(str)
    return features, labels, entity_ids


def stratified_split(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/validation split."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_valid, y_train, y_valid


def make_class_weights(labels: pd.Series) -> Dict[int, float]:
    """Return class weights balanced to the inverse of class frequency."""
    value_counts = labels.value_counts()
    total = len(labels)
    weights = {int(cls): total / (len(value_counts) * count) for cls, count in value_counts.items()}
    return weights
