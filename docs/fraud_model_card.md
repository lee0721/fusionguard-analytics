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

### Reproduction Commands
From the project root (`~/fusionguard-analytics`) on the HPC allocation:

```bash
# Request GPU session beforehand, e.g.
# srun --pty --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 bash

python -m src.models.fraud.train_xgboost \
  --feature-store data/feature_store.parquet \
  --output-dir artifacts/fraud/xgboost

python -m src.models.fraud.train_autoencoder \
  --feature-store data/feature_store.parquet \
  --output-dir artifacts/fraud/autoencoder \
  --device cuda

python -m src.models.fraud.explain_xgboost \
  --feature-store data/feature_store.parquet \
  --model-path artifacts/fraud/xgboost/xgboost_model.json \
  --output-dir docs/assets/fraud
```

## Evaluation Summary

| Model       | Precision | Recall | F1    | ROC AUC | Average Precision (AUCPR) |
|-------------|-----------|--------|-------|---------|---------------------------|
| XGBoost     | 0.8817    | 0.8367 | 0.8586| 0.9768  | 0.8797                    |
| Autoencoder | 0.0539    | 0.8571 | 0.1015| 0.9617  | 0.6433                    |

**Observations**
- Gradient-boosted trees achieve strong balance between precision and recall with AUCPR near 0.88, suitable for precision-critical alerting.
- The autoencoder captures the majority of fraud cases (recall ~0.86) but with low precision; useful as a high-recall filter feeding a second-stage model.

## Explainability
- Generate SHAP plots: `python src/models/fraud/explain_xgboost.py`
- Outputs stored under `docs/assets/fraud/`:
  - `xgboost_shap_importance.png`
  - `xgboost_shap_summary.png`
  - `xgboost_shap_importance.csv`
- Top ranked SHAP features (|mean SHAP|): V14, V4, V12, V10, V3, reflecting strong influence of anonymised PCA components on predictions.

## Extended Evaluation
- Run the extended evaluation utility to derive calibration plots and threshold recommendations:
  ```bash
  python -m src.models.fraud.evaluate_extended \
    --xgb-predictions artifacts/fraud/xgboost/validation_predictions.csv \
    --autoencoder-scores artifacts/fraud/autoencoder/validation_scores.csv \
    --output-dir artifacts/fraud/evaluation
  ```
- Generated assets:
  - `artifacts/fraud/evaluation/xgboost_calibration.png`, `autoencoder_calibration.png`
  - Threshold sweep tables (`xgboost_threshold_metrics.csv`, `autoencoder_threshold_metrics.csv`)
  - `artifacts/fraud/evaluation/evaluation_extended.json` summarising ROC/AUCPR and best threshold settings.
- Incorporate the recommended thresholds into alerting configs and MLOps pipeline to align with business budgets.

## Operational Notes
- **Class Imbalance:** Highly skewed; `scale_pos_weight` and AUCPR are critical metrics.
- **Thresholding:** Autoencoder threshold defaults to the 97.5th percentile of training reconstruction error (override with `--threshold-percentile`).
- **Acceleration:** Autoencoder supports GPU when available (`--device cuda`).
- **Reproducibility:** Ensure feature store is regenerated on target hardware before training; random seeds default to 42.

## Next Steps
- Compare deployment cost/latency between XGBoost and autoencoder.
- Integrate both outputs into `mlops/train_pipeline.py` for scheduled retraining.
- Extend evaluation with calibration plots and threshold-optimization for business-defined alert budgets.
