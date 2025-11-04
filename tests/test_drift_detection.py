from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift_detection import (
    compute_drift_report,
    compute_population_stability_index,
)


def test_population_stability_index_identical_samples():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0] * 50, name="feature")
    current = reference.copy()
    psi = compute_population_stability_index(reference, current)
    assert psi < 1e-6


def test_drift_report_flags_numeric_shift():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"feature": rng.normal(loc=0.0, scale=1.0, size=1000)})
    current = pd.DataFrame({"feature": rng.normal(loc=1.5, scale=1.0, size=1000)})

    report = compute_drift_report(
        reference=reference,
        current=current,
        features=["feature"],
        psi_threshold=0.1,
        ks_pvalue_threshold=0.05,
    )

    assert report.feature_results[0].has_alert
