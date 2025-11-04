"""Streamlit dashboard for monitoring data and performance drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_detection import compute_drift_report, compute_performance_drift


@st.cache_data(show_spinner=False)
def load_feature_store(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "features" in df.columns:
        feature_values = pd.json_normalize(df["features"])
        feature_values.columns = [str(col) for col in feature_values.columns]
        df = pd.concat([df.drop(columns=["features"]), feature_values], axis=1)
    return df


def load_baseline_metrics(artifact_dir: Path) -> Dict[str, float]:
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as fh:
            metrics = json.load(fh)
            return {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
    return {}


def inject_gaussian_drift(df: pd.DataFrame, feature_names: List[str], *, strength: float, seed: int) -> pd.DataFrame:
    if strength <= 0:
        return df.copy()

    rng = np.random.default_rng(seed)
    drifted = df.copy()
    for feature in feature_names:
        if feature not in drifted.columns:
            continue
        series = drifted[feature]
        if pd.api.types.is_numeric_dtype(series):
            scale = series.std(ddof=0) or 1.0
            drifted[feature] = series + rng.normal(loc=0.0, scale=scale * strength, size=len(series))
    return drifted


def app() -> None:
    st.set_page_config(page_title="FusionGuard Monitoring", layout="wide")
    st.title("FusionGuard Drift Monitoring Dashboard")

    feature_store_path = Path(
        st.sidebar.text_input(
            "Feature store parquet",
            value="data/feature_store.parquet",
        )
    )

    if not feature_store_path.exists():
        st.error(f"Feature store not found at {feature_store_path}")
        st.stop()

    df = load_feature_store(feature_store_path)
    st.sidebar.caption(f"Columns loaded: {len(df.columns)}")
    st.sidebar.write(df.columns.tolist()[:10])
    available_datasets = sorted(df["dataset"].unique()) if "dataset" in df.columns else ["all"]
    dataset_choice = st.sidebar.selectbox("Dataset segment", options=available_datasets, index=0)

    if dataset_choice != "all" and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_choice]

    if df.empty:
        st.warning("No rows available for the selected segment. Please choose another subset or broaden the filter.")
        st.stop()

    feature_columns = [col for col in df.columns if col not in {"entity_id", "dataset", "label"}]
    numeric_features = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    default_pool = numeric_features or categorical_features or feature_columns
    default_features = default_pool[: min(5, len(default_pool))]
    options = numeric_features + categorical_features
    if not options:
        options = feature_columns

    monitored_features = st.sidebar.multiselect(
        "Monitored Features",
        options=options,
        default=default_features,
    )

    if not monitored_features:
        st.warning("Please select at least one feature.")
        st.stop()

    ref_fraction = st.sidebar.slider("Reference Fraction", min_value=0.1, max_value=0.9, value=0.5)
    random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    drift_strength = st.sidebar.slider("Inject Simulated Drift (0-strong)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    ref_sample = df.sample(frac=ref_fraction, random_state=int(random_seed))
    cur_sample = df.drop(ref_sample.index)
    if cur_sample.empty:
        cur_sample = df.sample(frac=1 - ref_fraction, random_state=int(random_seed) + 1)

    cur_sample = inject_gaussian_drift(cur_sample, monitored_features, strength=drift_strength, seed=int(random_seed))

    psi_threshold = st.sidebar.number_input("PSI Threshold", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
    ks_threshold = st.sidebar.number_input("KS Test p-value Threshold", value=0.05, min_value=0.0, max_value=0.5, step=0.01)

    drift_report = compute_drift_report(
        reference=ref_sample,
        current=cur_sample,
        features=monitored_features,
        psi_threshold=psi_threshold,
        ks_pvalue_threshold=ks_threshold,
    )

    st.subheader("Data Drift Detection")
    df_results = pd.DataFrame(
        [
            {
                "feature": result.feature,
                "psi": result.psi,
                "ks_statistic": result.ks_statistic,
                "ks_pvalue": result.ks_pvalue,
                "alerts": ", ".join(result.alerts) if result.alerts else "OK",
            }
            for result in drift_report.feature_results
        ]
    )
    st.dataframe(df_results, use_container_width=True)

    st.markdown(f"**Data drift alert:** {'🚨 triggered' if drift_report.data_alert else '✅ none'}")

    baseline_dir = Path(st.sidebar.text_input("Baseline metrics path", value="artifacts/fraud/xgboost"))
    baseline_metrics = load_baseline_metrics(baseline_dir)

    st.subheader("Performance Drift Simulation")
    perf_inputs: Dict[str, float] = {}
    tolerances: Dict[str, float] = {}

    for metric, value in baseline_metrics.items():
        st.markdown(f"- **{metric}** baseline: {value:.3f}")
        tolerances[metric] = st.sidebar.number_input(f"{metric} tolerance", value=0.02, min_value=0.0, max_value=0.2, step=0.01)
        perf_inputs[metric] = st.slider(
            f"{metric} current value",
            min_value=0.0,
            max_value=1.0,
            value=float(value),
            step=0.01,
        )

    performance_results = compute_performance_drift(
        baseline_metrics=baseline_metrics,
        current_metrics=perf_inputs,
        tolerances=tolerances,
    )

    if performance_results:
        perf_df = pd.DataFrame(
            [
                {
                    "metric": res.metric,
                    "baseline": res.baseline,
                    "current": res.current,
                    "delta": res.delta,
                    "tolerance": res.tolerance,
                    "alert": "🚨" if res.alert else "✅",
                }
                for res in performance_results
            ]
        )
        st.dataframe(perf_df, use_container_width=True)

        perf_alert = any(res.alert for res in performance_results)
        st.markdown(f"**Performance drift alert:** {'🚨 triggered' if perf_alert else '✅ none'}")
    else:
        st.info("Baseline metrics not found. Point the sidebar path to a folder containing `metrics.json`.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Launch with `streamlit run src/monitoring/dashboard.py`.")


if __name__ == "__main__":
    app()
