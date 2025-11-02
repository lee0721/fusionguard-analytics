#!/usr/bin/env python3
"""Benchmark inference speed for fraud detection models."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from numpy.typing import ArrayLike

from .data_utils import load_fraud_dataset
from .train_autoencoder import FraudAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark XGBoost and Autoencoder inference latency.")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to consolidated feature store parquet.",
    )
    parser.add_argument(
        "--xgb-model",
        type=Path,
        default=Path("artifacts/fraud/xgboost/xgboost_model.json"),
        help="Path to trained XGBoost model (JSON).",
    )
    parser.add_argument(
        "--autoencoder-artifact",
        type=Path,
        default=Path("artifacts/fraud/autoencoder/autoencoder.pt"),
        help="PyTorch checkpoint produced by train_autoencoder.",
    )
    parser.add_argument(
        "--autoencoder-scaler",
        type=Path,
        default=Path("artifacts/fraud/autoencoder/scaler.pkl"),
        help="StandardScaler used for autoencoder training.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of fraud feature rows to sample for benchmarking.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs to average for each model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to benchmark autoencoder on (XGBoost always runs on CPU).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fraud/benchmark/results.json"),
        help="Where to store benchmark results JSON.",
    )
    return parser.parse_args()


def load_sample(features: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    if sample_size > len(features):
        raise ValueError(f"Sample size {sample_size} exceeds dataset size {len(features)}.")
    return features.sample(sample_size, random_state=random_state).reset_index(drop=True)


def load_xgb_model(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def run_xgb_inference(booster: xgb.Booster, batch: pd.DataFrame) -> np.ndarray:
    dmatrix = xgb.DMatrix(batch)
    return booster.predict(dmatrix)


def load_autoencoder(artifact_path: Path, device: torch.device) -> Tuple[FraudAutoencoder, Dict[str, float]]:
    checkpoint = torch.load(artifact_path, map_location=device)
    model = FraudAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, {"threshold": checkpoint.get("threshold", None)}


def run_autoencoder_inference(
    model: FraudAutoencoder, data: torch.Tensor, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        reconstructed = model(data.to(device))
        errors = torch.mean((reconstructed - data.to(device)) ** 2, dim=1)
    return errors.cpu().numpy()


def measure_latency(fn, *args, runs: int) -> Dict[str, float]:
    timings = []
    # Warm-up
    fn(*args)
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args)
        timings.append(time.perf_counter() - start)
    return {
        "runs": runs,
        "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings),
        "std_seconds": statistics.pstdev(timings) if runs > 1 else 0.0,
        "per_run_seconds": timings,
    }


def benchmark(args: argparse.Namespace) -> Dict[str, object]:
    features, _, _ = load_fraud_dataset(args.feature_store)
    sample = load_sample(features, args.sample_size)

    results: Dict[str, object] = {
        "sample_size": args.sample_size,
        "runs": args.runs,
        "device": args.device,
    }

    # XGBoost benchmark (CPU)
    xgb_model = load_xgb_model(args.xgb_model)
    xgb_metrics = measure_latency(run_xgb_inference, xgb_model, sample, runs=args.runs)
    xgb_metrics["throughput_per_second"] = args.sample_size / xgb_metrics["mean_seconds"]
    results["xgboost"] = xgb_metrics

    # Autoencoder benchmark
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("⚠️ CUDA requested but not available; benchmarking autoencoder on CPU.")
    scaler = joblib.load(args.autoencoder_scaler)
    scaled_sample = scaler.transform(sample)
    tensor = torch.tensor(scaled_sample, dtype=torch.float32, device=device)

    autoencoder, metadata = load_autoencoder(args.autoencoder_artifact, device)
    ae_metrics = measure_latency(run_autoencoder_inference, autoencoder, tensor, device, runs=args.runs)
    ae_metrics["throughput_per_second"] = args.sample_size / ae_metrics["mean_seconds"]
    ae_metrics["metadata"] = metadata
    results["autoencoder"] = ae_metrics

    return results


def main() -> None:
    args = parse_args()
    results = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"✅ Benchmark results saved to {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
