#!/usr/bin/env python3
"""Shared utilities for loading and preparing the fraud feature store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_feature_store(feature_store_path: Path | str) -> pd.DataFrame:
    """Load the consolidated feature store parquet into a DataFrame."""
    path = Path(feature_store_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found at {path}")
    return pd.read_parquet(path)


def load_fraud_dataset(
    feature_store_path: Path | str,
    dataset_name: str = "fraud",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return feature matrix, label vector, and entity ids for the fraud dataset."""
    raw_df = load_feature_store(feature_store_path)
    filtered = raw_df[raw_df["dataset"] == dataset_name].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for dataset='{dataset_name}' in feature store.")

    features = pd.json_normalize(filtered["features"].map(json.loads))
    features.columns = [col.replace(".", "_") for col in features.columns]
    labels = filtered["label"].astype(int)
    entity_ids = filtered["entity_id"]
    return features, labels, entity_ids


def stratified_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a stratified train/validation split."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_valid, y_train, y_valid
