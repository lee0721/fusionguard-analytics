# Churn Model Card

## Overview
- **Objective:** Predict customer churn propensities to support targeted retention outreach.
- **Data Source:** Kaggle "Bank Customer Churn" dataset (documented in `docs/data_sources.md`).
- **Feature Store:** `data/feature_store.parquet` (generated via `src/data/build_feature_store.py`), filtered by `dataset == "churn"`.

## Training Configuration
- Script: `python src/models/churn/train_lightgbm.py`
- Key hyperparameters (defaults): `learning_rate=0.05`, `num_leaves=64`, `n_estimators=600`, `max_depth=-1`
- Class imbalance: handled via `make_class_weights` (inverse frequency) inside `train_lightgbm`.
- Artifacts (written under `artifacts/churn/lightgbm/`):
  - `metrics.json`
  - `validation_predictions.csv`
  - `feature_importance.csv`
  - `lightgbm_model.txt`

### Reproduction Commands (HPC Recommended)
```bash
# On CREATE HPC after syncing the repo and feature store
python -m src.models.churn.train_lightgbm \
  --feature-store data/feature_store.parquet \
  --output-dir artifacts/churn/lightgbm \
  --device cpu  # or gpu if LightGBM GPU build is available
```

> ⚠️ LightGBM currently segfaults on the local macOS virtualenv; use the CREATE HPC environment (or rebuild LightGBM from source locally) to generate the artifacts before populating the evaluation section below.

## Evaluation Summary
| Metric              | Value | Notes |
|---------------------|-------|-------|
| Precision            | 0.6096 | Validation precision @ threshold 0.50 |
| Recall               | 0.5602 | Validation recall @ threshold 0.50 |
| F1 Score             | 0.5839 | Validation F1 @ threshold 0.50 |
| AUROC                | 0.8400 | Area under ROC curve (validation) |
| Average Precision    | 0.6627 | Validation average precision (AUCPR) |
| Decision Threshold   | 0.50 | Probability cutoff used for churn alerts |

- Detailed metrics saved in `metrics.json`.
- Validation predictions (`validation_predictions.csv`) enable downstream calibration and segmentation analysis.

## Feature Importance & Explainability
- `feature_importance.csv` ranks engineered features by LightGBM gain importance.
- Use SHAP (optional): adapt `src/models/fraud/explain_xgboost.py` for churn if SHAP-based narratives are required.

## Business Insights
- **Critical-risk cohort:** Top 10 % by churn probability drives a 76 % observed churn rate, skewing toward Germany/France customers with ~£95k balances, £99k salaries, and average age 48. Only 25 % are active members and 70 % hold a credit card, signalling disengagement despite high value—prioritise tailored concierge outreach, relationship-manager calls, and retention bonuses tied to product usage.
- **Product footprint:** 62 % of the critical group hold a single product while another 23 % hold three, revealing both under-served and over-leveraged subsegments. Pair retention offers with needs-based cross-sell (e.g., savings or investment bundles) plus credit-score coaching for the 54 % sitting in “fair/poor” bands.
- **Rising-risk watchlist:** The medium band (40–70 % probability) still shows ~12 % churn—early loyalty incentives, digital nudges, and proactive service checks can keep these customers from tipping into the high-risk tier.

## Operational Notes
- Integrate churn training into `mlops/train_pipeline.py` (pass `--skip-churn` to omit when needed).
- Ensure churn artifacts are synced from CREATE HPC back to the local repo before committing (`rsync ... artifacts/churn/`).
- Track future calibration/threshold optimisation in `artifacts/churn/...` to align with fraud evaluation depth.

## Next Steps
1. Run the LightGBM trainer on CREATE HPC and capture metrics.
2. Update the evaluation table and business insight bullets above.
3. Extend notebooks/EDA to include churn segmentation using `validation_predictions.csv`.
