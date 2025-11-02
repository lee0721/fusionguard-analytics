#!/usr/bin/env python3
"""Train a PyTorch autoencoder for credit card fraud anomaly detection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data_utils import load_fraud_dataset, stratified_split


class FraudAutoencoder(nn.Module):
    """Simple feed-forward autoencoder for tabular anomaly detection."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoencoder for fraud detection.")
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path to consolidated feature store.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fraud/autoencoder"),
        help="Directory to store trained model and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent representation size.")
    parser.add_argument("--threshold-percentile", type=float, default=97.5, help="Percentile of training reconstruction error used as anomaly threshold.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda or cpu).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruction_errors(model: FraudAutoencoder, data: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        recon = model(data)
        errors = torch.mean((recon - data) ** 2, dim=1)
    return errors.cpu().numpy()


def compute_metrics(errors: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, object]:
    preds = (errors >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, errors),
        "average_precision": average_precision_score(labels, errors),
    }
    precision, recall, pr_thresholds = precision_recall_curve(labels, errors)
    metrics["precision_recall_curve"] = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist(),
    }
    return metrics


def train_autoencoder(
    feature_store: Path,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    threshold_percentile: float = 97.5,
    random_state: int = 42,
    device: str | None = None,
) -> Dict[str, object]:
    """Train autoencoder and persist artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(random_state)

    features, labels, entity_ids = load_fraud_dataset(feature_store)
    entity_series = entity_ids.rename("entity_id")

    X_train_full, X_valid, y_train_full, y_valid = stratified_split(
        features, labels, test_size=0.2, random_state=random_state
    )
    entity_valid = entity_series.iloc[X_valid.index.to_numpy()]

    y_train_array = y_train_full.to_numpy()
    legit_mask = y_train_array == 0
    X_train_legit = X_train_full.iloc[legit_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_legit)
    X_valid_scaled = scaler.transform(X_valid)

    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(requested_device)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = FraudAutoencoder(
        input_dim=X_train_scaled.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(torch_device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(torch_device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = float(np.mean(batch_losses))
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch:03d} | loss={epoch_loss:.6f}")

    train_errors = reconstruction_errors(model, train_tensor.to(torch_device))
    threshold = float(np.percentile(train_errors, threshold_percentile))

    valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32).to(torch_device)
    valid_errors = reconstruction_errors(model, valid_tensor)

    y_valid_array = y_valid.to_numpy()
    metrics = compute_metrics(valid_errors, y_valid_array, threshold)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    recon_df = pd.DataFrame(
        {
            "entity_id": entity_valid.values,
            "label": y_valid_array,
            "reconstruction_error": valid_errors,
            "prediction": (valid_errors >= threshold).astype(int),
        }
    )
    validation_scores_path = output_dir / "validation_scores.csv"
    recon_df.to_csv(validation_scores_path, index=False)

    model_artifact = {
        "input_dim": X_train_scaled.shape[1],
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "state_dict": model.state_dict(),
        "threshold": threshold,
    }
    model_path = output_dir / "autoencoder.pt"
    torch.save(model_artifact, model_path)

    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    return {
        "metrics": metrics,
        "history": history,
        "artifact_paths": {
            "metrics": metrics_path,
            "history": history_path,
            "validation_scores": validation_scores_path,
            "model": model_path,
            "scaler": scaler_path,
        },
        "threshold": threshold,
        "device": torch_device.type,
    }


def main() -> None:
    args = parse_args()
    result = train_autoencoder(
        feature_store=args.feature_store,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        threshold_percentile=args.threshold_percentile,
        random_state=args.random_state,
        device=args.device,
    )
    paths = result["artifact_paths"]
    print(f"✅ Metrics saved to {paths['metrics']}")
    print(f"✅ Model saved to {paths['model']}")


if __name__ == "__main__":
    main()
