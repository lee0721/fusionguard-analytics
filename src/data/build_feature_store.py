#!/usr/bin/env python3
"""
Build processed datasets and a reusable feature store for the FusionGuard analytics project.

This script reads the raw Kaggle datasets (credit card fraud and bank customer churn),
performs lightweight cleaning and feature engineering with PySpark, writes cleaned
parquet datasets to ``data/processed``, combines the curated features into
``data/feature_store.parquet``, and emits simple data bias diagnostics (class imbalance).

Example:

    python src/data/build_feature_store.py \\
        --raw-dir data/raw \\
        --processed-dir data/processed \\
        --feature-store data/feature_store.parquet \\
        --bias-report data/processed/data_bias_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from pyspark.sql import DataFrame, SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets and feature store.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw Kaggle datasets.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for cleaned parquet datasets.",
    )
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data/feature_store.parquet"),
        help="Path (directory) for the consolidated feature store parquet.",
    )
    parser.add_argument(
        "--bias-report",
        type=Path,
        default=Path("data/processed/data_bias_report.json"),
        help="Path to write class imbalance diagnostics (JSON).",
    )
    return parser.parse_args()


def build_spark(app_name: str = "FusionGuardFeatureBuilder") -> SparkSession:
    """Initialise a local Spark session suitable for lightweight data prep."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def _credit_score_bucket() -> F.Column:
    """Utility expression for credit score banding."""
    return (
        F.when(F.col("CreditScore") >= 750, F.lit("excellent"))
        .when(F.col("CreditScore") >= 650, F.lit("good"))
        .when(F.col("CreditScore") >= 550, F.lit("fair"))
        .otherwise(F.lit("poor"))
    )


def process_churn_dataset(spark: SparkSession, raw_dir: Path) -> Tuple[DataFrame, DataFrame]:
    """Read and engineer features for the bank customer churn dataset."""
    source = raw_dir / "bank_churn" / "Churn_Modelling.csv"
    df = spark.read.csv(str(source), header=True, inferSchema=True)

    cleaned = (
        df.drop("RowNumber", "Surname")
        .withColumnRenamed("CustomerId", "customer_id")
        .withColumnRenamed("Exited", "label")
        .withColumn("is_active_member", F.col("IsActiveMember").cast("int"))
        .withColumn("has_cr_card", F.col("HasCrCard").cast("int"))
        .withColumn("balance_salary_ratio", F.when(F.col("EstimatedSalary") > 0, F.col("Balance") / F.col("EstimatedSalary")).otherwise(F.lit(0.0)))
        .withColumn("credit_score_bucket", _credit_score_bucket())
        .withColumn(
            "tenure_bucket",
            F.when(F.col("Tenure") < 3, F.lit("short"))
            .when(F.col("Tenure") < 7, F.lit("medium"))
            .otherwise(F.lit("long")),
        )
    )

    feature_cols = [
        F.col("Geography"),
        F.col("Gender"),
        F.col("Age"),
        F.col("Tenure"),
        F.col("tenure_bucket"),
        F.col("Balance"),
        F.col("EstimatedSalary"),
        F.col("NumOfProducts"),
        F.col("IsActiveMember").alias("is_active_flag"),
        F.col("is_active_member"),
        F.col("has_cr_card"),
        F.col("CreditScore"),
        F.col("credit_score_bucket"),
        F.col("balance_salary_ratio"),
    ]

    features = (
        cleaned.select(
            F.col("customer_id"),
            F.col("label"),
            F.struct(*feature_cols).alias("feature_struct"),
        )
        .withColumn("entity_id", F.col("customer_id").cast("string"))
        .withColumn("dataset", F.lit("churn"))
        .withColumn("features", F.to_json("feature_struct"))
        .select("entity_id", "dataset", "features", "label")
    )

    return cleaned, features


def process_fraud_dataset(spark: SparkSession, raw_dir: Path) -> Tuple[DataFrame, DataFrame]:
    """Read and engineer features for the credit card fraud dataset."""
    source = raw_dir / "creditcard_fraud" / "creditcard.csv"
    df = spark.read.csv(str(source), header=True, inferSchema=True)

    cleaned = (
        df.withColumnRenamed("Class", "label")
        .withColumn("transaction_id", F.monotonically_increasing_id())
        .withColumn("amount_log", F.log1p(F.col("Amount")))
        .withColumn("time_hours", (F.col("Time") / F.lit(3600)).cast("int"))
        .withColumn(
            "is_night",
            F.when((F.col("time_hours") >= 0) & (F.col("time_hours") < 6), F.lit(1)).otherwise(F.lit(0)),
        )
    )

    v_cols = [F.col(f"V{i}") for i in range(1, 29)]
    feature_cols = [
        F.col("Amount"),
        F.col("amount_log"),
        F.col("time_hours"),
        F.col("is_night"),
        *v_cols,
    ]

    features = (
        cleaned.select(
            F.col("transaction_id"),
            F.col("label"),
            F.struct(*feature_cols).alias("feature_struct"),
        )
        .withColumn("entity_id", F.col("transaction_id").cast("string"))
        .withColumn("dataset", F.lit("fraud"))
        .withColumn("features", F.to_json("feature_struct"))
        .select("entity_id", "dataset", "features", "label")
    )

    return cleaned, features


def calculate_class_balance(df: DataFrame, label_col: str) -> Dict[str, float]:
    """Return class distribution as a ratio dictionary."""
    counts = df.groupBy(label_col).count()
    total = counts.agg(F.sum("count").alias("total")).collect()[0]["total"]
    summary = {str(row[label_col]): row["count"] / total for row in counts.collect()}
    return summary


def main() -> None:
    args = parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.feature_store.parent.mkdir(parents=True, exist_ok=True)
    args.bias_report.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    try:
        churn_cleaned, churn_features = process_churn_dataset(spark, args.raw_dir)
        fraud_cleaned, fraud_features = process_fraud_dataset(spark, args.raw_dir)

        churn_output = args.processed_dir / "bank_churn"
        fraud_output = args.processed_dir / "creditcard_fraud"

        churn_cleaned.write.mode("overwrite").parquet(str(churn_output))
        fraud_cleaned.write.mode("overwrite").parquet(str(fraud_output))

        feature_store = churn_features.unionByName(fraud_features, allowMissingColumns=True)
        feature_store.write.mode("overwrite").parquet(str(args.feature_store))

        bias_metrics = {
            "churn": calculate_class_balance(churn_cleaned, "label"),
            "fraud": calculate_class_balance(fraud_cleaned, "label"),
        }

        args.bias_report.write_text(json.dumps(bias_metrics, indent=2))

        print("✅ Processed datasets written to:", churn_output, "and", fraud_output)
        print("✅ Feature store saved to:", args.feature_store)
        print("✅ Bias report saved to:", args.bias_report)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
