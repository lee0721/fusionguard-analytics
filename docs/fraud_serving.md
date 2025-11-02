# Fraud Model Serving & Benchmarking

## Benchmark Procedure

Run the inference benchmark utility after training artifacts are available:

```bash
# CPU baseline
python -m src.models.fraud.benchmark_inference \
  --device cpu \
  --sample-size 10000 \
  --runs 5 \
  --output artifacts/fraud/benchmark/results_cpu.json

# GPU-accelerated autoencoder (inside GPU allocation)
python -m src.models.fraud.benchmark_inference \
  --device cuda \
  --sample-size 10000 \
  --runs 5 \
  --output artifacts/fraud/benchmark/results_gpu.json
```

Both runs load the feature store, perform warm-up, and report average latency along with throughput (predictions per second). Results are stored as JSON for downstream cost modelling.

## Latest Benchmark Snapshot

| Model / Device | Mean Latency (ms) | Median Latency (ms) | Throughput (pred/s) | Notes |
|----------------|------------------|---------------------|---------------------|-------|
| XGBoost (CPU)  | _pending_        | _pending_           | _pending_           | Single-thread hist tree |
| Autoencoder (CPU) | _pending_     | _pending_           | _pending_           | CPU fallback baseline |
| Autoencoder (GPU) | _pending_     | _pending_           | _pending_           | Measured on CREATE GPU node |

> Update the table after running the commands above by reading the JSON files and converting seconds to milliseconds (`ms = seconds * 1000`). A helper snippet:

```bash
python - <<'PY'
import json, pathlib
for name in ["results_cpu.json", "results_gpu.json"]:
    path = pathlib.Path("artifacts/fraud/benchmark") / name
    if path.exists():
        data = json.loads(path.read_text())
        ae = data.get("autoencoder")
        xgb = data.get("xgboost")
        print(name, "-> autoencoder:", ae["mean_seconds"], "xgboost:", xgb["mean_seconds"])
PY
```

## Cost Estimation Methodology

- **CPU baseline:** assume Cloud Run or GKE autopilot at roughly USD $0.000024 per vCPU-second. Cost per 1M predictions ≈ `(1 / throughput) * 1e6 * vCPU_rate`.
- **GPU batch scoring:** assume Nvidia A10G on GCP (≈ USD $0.004 per GPU-second). Cost per 1M predictions ≈ `(1 / throughput_gpu) * 1e6 * gpu_rate`. Autoencoder on GPU is most beneficial for large batch offline scoring.
- Include additional overhead for memory and networking (5–10% buffer).

## Reporting

Document key figures inside `docs/fraud_model_card.md` (serving section) once benchmarks are collected. Include:

- Latency & throughput comparison.
- Break-even point where GPU overtakes CPU in cost-per-1M predictions.
- Deployment recommendation (e.g., XGBoost for online scoring, autoencoder GPU for nightly batch anomaly sweeps).
