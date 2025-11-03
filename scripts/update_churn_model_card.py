#!/usr/bin/env python3
"""Update churn model card metrics table from the latest metrics JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate docs/churn_model_card.md with metrics from metrics.json."
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/churn/lightgbm/metrics.json"),
        help="Path to the churn LightGBM metrics.json file.",
    )
    parser.add_argument(
        "--model-card-path",
        type=Path,
        default=Path("docs/churn_model_card.md"),
        help="Path to the churn model card markdown file.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to display for metric values.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {
        "threshold": float(data["threshold"]),
        "precision": float(data["precision"]),
        "recall": float(data["recall"]),
        "f1": float(data["f1"]),
        "roc_auc": float(data["roc_auc"]),
        "average_precision": float(data["average_precision"]),
    }


def format_value(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if digits >= 0 else str(value)


def build_updates(metrics: dict[str, float], digits: int) -> dict[str, tuple[str, str]]:
    threshold = metrics["threshold"]
    return {
        "Precision": (
            format_value(metrics["precision"], digits),
            f"Validation precision @ threshold {format_value(threshold, 2)}",
        ),
        "Recall": (
            format_value(metrics["recall"], digits),
            f"Validation recall @ threshold {format_value(threshold, 2)}",
        ),
        "F1 Score": (
            format_value(metrics["f1"], digits),
            f"Validation F1 @ threshold {format_value(threshold, 2)}",
        ),
        "AUROC": (
            format_value(metrics["roc_auc"], digits),
            "Area under ROC curve (validation)",
        ),
        "Average Precision": (
            format_value(metrics["average_precision"], digits),
            "Validation average precision (AUCPR)",
        ),
        "Decision Threshold": (
            format_value(threshold, 2),
            "Probability cutoff used for churn alerts",
        ),
    }


def update_table(markdown: str, updates: dict[str, tuple[str, str]]) -> str:
    updated = markdown
    for label, (value, note) in updates.items():
        pattern = re.compile(rf"^\| {re.escape(label)}\s*\|.*$", re.MULTILINE)
        if not pattern.search(updated):
            raise ValueError(f"Could not find table row for '{label}' in model card.")
        name_cell = label.ljust(20)
        line = f"| {name_cell} | {value} | {note} |"
        updated = pattern.sub(line, updated, count=1)
    return updated


def main() -> None:
    args = parse_args()
    if not args.metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {args.metrics_path}")
    if not args.model_card_path.exists():
        raise FileNotFoundError(f"Model card not found at {args.model_card_path}")

    metrics = load_metrics(args.metrics_path)
    updates = build_updates(metrics, args.precision)

    markdown = args.model_card_path.read_text()
    updated_markdown = update_table(markdown, updates)
    args.model_card_path.write_text(updated_markdown)
    print(f"✅ Updated {args.model_card_path} with metrics from {args.metrics_path}")


if __name__ == "__main__":
    main()
