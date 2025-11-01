# Fraud Detection Model Card

## Overview
- **Objective:** Detect anomalous credit card transactions using supervised (XGBoost) and unsupervised (autoencoder) approaches.
- **Data Source:** Kaggle “Credit Card Fraud Detection” dataset (see `docs/data_sources.md`).
- **Feature Store:** `data/feature_store.parquet` (generated via `src/data/build_feature_store.py`).

## Training Configuration
### XGBoost Classifier
- Script: `python src/models/fraud/train_xgboost.py`
- Key hyperparameters: `n_estimators=600`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight` auto-derived from class imbalance.
- Artifacts:
  - Model: `artifacts/fraud/xgboost/xgboost_model.json`
  - Metrics: `artifacts/fraud/xgboost/metrics.json`
  - Validation predictions: `artifacts/fraud/xgboost/validation_predictions.csv`
  - Feature importance: `artifacts/fraud/xgboost/feature_importance.csv`

### Autoencoder (PyTorch)
- Script: `python src/models/fraud/train_autoencoder.py`
- Training strategy: Train on legitimate transactions only, detect anomalies via reconstruction error.
- Key hyperparameters: `hidden_dim=128`, `latent_dim=32`, `epochs=50`, `batch_size=4096`.
- Artifacts:
  - Model checkpoint: `artifacts/fraud/autoencoder/autoencoder.pt`
  - Scaler: `artifacts/fraud/autoencoder/scaler.pkl`
  - Metrics: `artifacts/fraud/autoencoder/metrics.json`
  - Validation scores: `artifacts/fraud/autoencoder/validation_scores.csv`

## Evaluation Summary
> Populate after running the training scripts.

| Model        | Precision | Recall | F1 | ROC AUC | Average Precision (AUCPR) |
|--------------|-----------|--------|----|---------|---------------------------|
| XGBoost      | TBD       | TBD    | TBD| TBD     | TBD                       |
| Autoencoder  | TBD       | TBD    | TBD| TBD     | TBD                       |

## Explainability
- Generate SHAP plots: `python src/models/fraud/explain_xgboost.py`
- Outputs stored under `docs/assets/fraud/`:
  - `xgboost_shap_importance.png`
  - `xgboost_shap_summary.png`
  - `xgboost_shap_importance.csv`

## Operational Notes
- **Class Imbalance:** Highly skewed; `scale_pos_weight` and AUCPR are critical metrics.
- **Thresholding:** Autoencoder threshold defaults to the 97.5th percentile of training reconstruction error (override with `--threshold-percentile`).
- **Acceleration:** Autoencoder supports GPU when available (`--device cuda`).
- **Reproducibility:** Ensure feature store is regenerated on target hardware before training; random seeds default to 42.

## Next Steps
- Compare deployment cost/latency between XGBoost and autoencoder.
- Integrate both outputs into `mlops/train_pipeline.py` for scheduled retraining.
- Extend evaluation with calibration plots and threshold-optimization for business-defined alert budgets.
