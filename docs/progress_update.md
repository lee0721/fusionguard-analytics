# Progress Update

## Completed Milestones

### Step 3 – Data Exploration & Cleaning
- Both EDA notebooks (`notebooks/eda/churn_eda.ipynb`, `notebooks/eda/fraud_eda.ipynb`) have been executed and saved with outputs.
- Corresponding HTML reports live in `notebooks/eda/reports/churn_eda.html` and `notebooks/eda/reports/fraud_eda.html` for quick review.
- Candidate data quality checks are documented in `docs/data_quality_checks.md` and ready to be ported into Great Expectations suites.

### Step 4 – Data Engineering
- Added `src/data/build_feature_store.py`, a PySpark pipeline that cleans the raw Kaggle datasets, engineers reusable features, writes parquet outputs under `data/processed/`, and materialises a consolidated `data/feature_store.parquet`.
- Example invocation (within the recommended virtual environment):
  ```bash
  source .venv_spark/bin/activate
  python src/data/build_feature_store.py \
    --raw-dir data/raw \
    --processed-dir data/processed \
    --feature-store data/feature_store.parquet \
    --bias-report data/processed/data_bias_report.json
  ```
- Execution emits `data/processed/data_bias_report.json`, capturing churn and fraud class imbalance metrics for bias monitoring or GE validation.

## Virtual Environment Notes
- The original `.venv` crashes (segmentation fault) when importing `numpy`/`pyspark`; avoid using it.
- `.venv_spark` (Python 3.13) already contains `pyspark`, `pandas`, and `numpy`. Activate with `source .venv_spark/bin/activate` before running pipelines.
- If you prefer the system `python3`, ensure `pyspark` is installed via `python3 -m pip install --user pyspark` and that `PYTHONPATH` includes `$(python3 -m site --user-site)`.
