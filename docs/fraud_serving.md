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

| Model / Device      | Mean Latency (ms) | Median Latency (ms) | Throughput (pred/s) | Cost / 1M preds* | Notes |
|---------------------|------------------|---------------------|---------------------|------------------|-------|
| XGBoost (CPU)       | 26.58            | 26.39               | 376,250             | ~$0.000064       | Single-thread hist tree |
| Autoencoder (CPU)   | 3.53             | 3.44                | 2,832,766           | ~$0.000009       | CPU fallback baseline |
| XGBoost (GPU)       | 22.06            | 22.00               | 453,301             | ~$0.0088         | CUDA inference on CREATE GPU node |
| Autoencoder (GPU)   | 0.30             | 0.28                | 32,806,786          | ~$0.00012        | CUDA-accelerated reconstruction error |

_\*Assumes USD 2.4 × 10⁻⁵ per vCPU-second and USD 4 × 10⁻³ per GPU-second (A10G estimate)._  
Latency values come from 10,000-record batches; cost per 1 M predictions uses `mean_seconds / 10_000 * 1_000_000 × rate`.

For convenience, convert the JSON outputs with:

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

- Latency & throughput highlights (CPU vs GPU).
- Platform cost comparison (per million predictions) to identify when GPU acceleration is justified.
- Deployment recommendation: XGBoost remains most economical on CPU for online scoring; autoencoder GPU mode is ideal for massive offline sweeps when ultra-low latency is required or when GPU capacity is already provisioned.
