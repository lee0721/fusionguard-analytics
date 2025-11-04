#!/usr/bin/env python3
"""Lightweight feature-store validation without Great Expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REQUIRED_DATASETS = {"fraud", "churn"}


def _record(checks: List[Dict[str, Any]], name: str, success: bool, details: str | None = None) -> None:
    entry: Dict[str, Any] = {"name": name, "success": bool(success)}
    if details:
        entry["details"] = details
    checks.append(entry)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover
            pass
    return str(obj)


def validate_feature_store(
    feature_store_path: Path,
    report_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate the consolidated feature store parquet file using pandas assertions."""

    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store not found at {feature_store_path}")

    df = pd.read_parquet(feature_store_path)
    checks: List[Dict[str, Any]] = []

    _record(checks, "row_count_positive", len(df) > 0, f"row_count={len(df)}")

    null_entity = df["entity_id"].isna().sum()
    _record(checks, "entity_id_not_null", null_entity == 0, f"null_count={null_entity}")

    unexpected_datasets = sorted(set(df["dataset"]) - REQUIRED_DATASETS)
    _record(
        checks,
        "dataset_values_valid",
        not unexpected_datasets,
        f"unexpected={unexpected_datasets}" if unexpected_datasets else None,
    )

    null_features = df["features"].isna().sum()
    _record(checks, "features_not_null", null_features == 0, f"null_count={null_features}")

    null_label = df["label"].isna().sum()
    _record(checks, "label_not_null", null_label == 0, f"null_count={null_label}")

    bad_labels = sorted(set(df["label"]) - {0, 1})
    _record(
        checks,
        "label_values_valid",
        not bad_labels,
        f"unexpected_labels={bad_labels}" if bad_labels else None,
    )

    duplicates = df.duplicated(subset=["entity_id", "dataset"]).sum()
    _record(checks, "entity_dataset_unique", duplicates == 0, f"duplicate_count={int(duplicates)}")

    success = all(check["success"] for check in checks)
    result = {"success": success, "checks": checks}

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, default=_json_default))

    if not success:
        raise ValueError("Feature store validation failed. See report for details.")

    return result


__all__ = ["validate_feature_store"]
