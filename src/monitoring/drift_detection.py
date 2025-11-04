"""Utilities for detecting data and performance drift across production runs.

The helpers in this module are intentionally lightweight so they can execute
inside scheduled jobs (Prefect, Airflow, cron) without pulling in heavyweight
monitoring services.  They are designed to operate on the feature store parquet
outputs produced by ``mlops/train_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

PSI_DEFAULT_THRESHOLD = 0.2
KS_PVALUE_DEFAULT_THRESHOLD = 0.05
PERFORMANCE_DEFAULT_TOLERANCE = 0.02


@dataclass
class FeatureDriftResult:
    """Summary of drift statistics for a single feature."""

    feature: str
    psi: Optional[float]
    ks_statistic: Optional[float]
    ks_pvalue: Optional[float]
    alerts: List[str] = field(default_factory=list)

    @property
    def has_alert(self) -> bool:
        return bool(self.alerts)


@dataclass
class PerformanceDriftResult:
    """Summary of model performance drift checks."""

    metric: str
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    tolerance: float
    alert: bool


@dataclass
class DriftReport:
    """Container for feature and performance drift checks."""

    generated_at: datetime
    feature_results: List[FeatureDriftResult]
    performance_results: List[PerformanceDriftResult]
    data_alert: bool
    performance_alert: bool
    metadata: Dict[str, Any]

    @property
    def has_alerts(self) -> bool:
        return self.data_alert or self.performance_alert


def _validate_inputs(reference: pd.DataFrame, current: pd.DataFrame, features: Iterable[str]) -> List[str]:
    for frame_name, frame in [("reference", reference), ("current", current)]:
        if frame.empty:
            raise ValueError(f"{frame_name} dataframe is empty.")
    missing = [feat for feat in features if feat not in reference.columns or feat not in current.columns]
    if missing:
        raise KeyError(f"Features not present in both dataframes: {missing}")
    return list(features)


def compute_population_stability_index(
    reference: Iterable[Any],
    current: Iterable[Any],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the Population Stability Index (PSI) between two samples."""

    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if ref.empty or cur.empty:
        return float("nan")

    if ptypes.is_numeric_dtype(ref) and ptypes.is_numeric_dtype(cur):
        quantiles = np.linspace(0, 1, bins + 1)
        edges = ref.quantile(quantiles).to_numpy()
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
        if edges.size <= 2:
            return 0.0
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
    else:
        categories = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = ref.value_counts(normalize=False).reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur.value_counts(normalize=False).reindex(categories, fill_value=0).to_numpy()

    ref_dist = ref_counts / (ref_counts.sum() + epsilon)
    cur_dist = cur_counts / (cur_counts.sum() + epsilon)

    psi = np.sum((ref_dist - cur_dist) * np.log((ref_dist + epsilon) / (cur_dist + epsilon)))
    return float(psi)


def _compute_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    psi_threshold: float,
    ks_pvalue_threshold: float,
) -> FeatureDriftResult:
    psi_value = compute_population_stability_index(reference, current)
    alerts: List[str] = []

    ks_stat: Optional[float] = None
    ks_pvalue: Optional[float] = None

    if not np.isnan(psi_value) and psi_value >= psi_threshold:
        alerts.append(f"PSI {psi_value:.3f} >= threshold {psi_threshold}")

    if ptypes.is_numeric_dtype(reference) and ptypes.is_numeric_dtype(current):
        if reference.nunique(dropna=True) > 1 or current.nunique(dropna=True) > 1:
            ks_stat, ks_pvalue = stats.ks_2samp(reference.dropna(), current.dropna())
            if ks_pvalue is not None and ks_pvalue <= ks_pvalue_threshold:
                alerts.append(f"KS p-value {ks_pvalue:.3f} <= threshold {ks_pvalue_threshold}")

    return FeatureDriftResult(
        feature=reference.name,
        psi=None if np.isnan(psi_value) else float(psi_value),
        ks_statistic=None if ks_stat is None else float(ks_stat),
        ks_pvalue=None if ks_pvalue is None else float(ks_pvalue),
        alerts=alerts,
    )


def compute_performance_drift(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    *,
    tolerances: Optional[Dict[str, float]] = None,
) -> List[PerformanceDriftResult]:
    """Compare production metrics against a baseline and flag deviations."""

    tolerances = tolerances or {}
    results: List[PerformanceDriftResult] = []

    metrics = set(baseline_metrics) | set(current_metrics)
    for metric in sorted(metrics):
        baseline = baseline_metrics.get(metric)
        current = current_metrics.get(metric)
        tolerance = tolerances.get(metric, PERFORMANCE_DEFAULT_TOLERANCE)

        if baseline is None or current is None:
            result = PerformanceDriftResult(
                metric=metric,
                baseline=baseline,
                current=current,
                delta=None,
                tolerance=tolerance,
                alert=True,
            )
            results.append(result)
            continue

        delta = current - baseline
        alert = abs(delta) > tolerance
        results.append(
            PerformanceDriftResult(
                metric=metric,
                baseline=float(baseline),
                current=float(current),
                delta=float(delta),
                tolerance=float(tolerance),
                alert=alert,
            )
        )

    return results


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: Optional[Iterable[str]] = None,
    psi_threshold: float = PSI_DEFAULT_THRESHOLD,
    ks_pvalue_threshold: float = KS_PVALUE_DEFAULT_THRESHOLD,
    baseline_metrics: Optional[Dict[str, float]] = None,
    current_metrics: Optional[Dict[str, float]] = None,
    metric_tolerances: Optional[Dict[str, float]] = None,
) -> DriftReport:
    """Generate a consolidated drift report for feature distributions and KPIs."""

    if features is None:
        features = [col for col in reference.columns if col in current.columns]

    feature_names = _validate_inputs(reference, current, features)

    feature_results = [
        _compute_feature_drift(reference[name], current[name], psi_threshold=psi_threshold, ks_pvalue_threshold=ks_pvalue_threshold)
        for name in feature_names
    ]

    performance_results: List[PerformanceDriftResult] = []
    if baseline_metrics or current_metrics:
        performance_results = compute_performance_drift(
            baseline_metrics=baseline_metrics or {},
            current_metrics=current_metrics or {},
            tolerances=metric_tolerances,
        )

    metadata = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_time_range": (
            reference.get("event_time").min() if "event_time" in reference.columns else None,
            reference.get("event_time").max() if "event_time" in reference.columns else None,
        ),
        "current_time_range": (
            current.get("event_time").min() if "event_time" in current.columns else None,
            current.get("event_time").max() if "event_time" in current.columns else None,
        ),
        "psi_threshold": psi_threshold,
        "ks_pvalue_threshold": ks_pvalue_threshold,
    }

    data_alert = any(result.has_alert for result in feature_results)
    performance_alert = any(result.alert for result in performance_results)

    return DriftReport(
        generated_at=datetime.utcnow(),
        feature_results=feature_results,
        performance_results=performance_results,
        data_alert=data_alert,
        performance_alert=performance_alert,
        metadata=metadata,
    )
