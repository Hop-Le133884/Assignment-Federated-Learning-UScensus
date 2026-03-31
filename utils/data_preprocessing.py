"""utils/data_preprocessing.py — Preprocess UCI Adult Income dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN, drop NaN rows, drop redundant columns."""
    df = df.replace("?", np.nan).dropna()
    # Drop fnlwgt (census weight, not predictive) and education text (use education_num)
    df = df.drop(columns=["fnlwgt", "education"])
    return df


def _encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Convert income string to binary label (1 = >50K)."""
    df = df.copy()
    df["label"] = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int)
    df = df.drop(columns=["income"])
    return df


def preprocess(input_path: str, output_path: str, stats_output_path: str = None):
    """
    Preprocess adult.data training file.

    Fits normalization stats and one-hot encoding on this data
    saves stats JSON for consistent transform of the train and test set.

    Returns (df_processed, stats_dict).
    """
    df = pd.read_csv(input_path, header=None, names=COLUMN_NAMES, skipinitialspace=True)
    df = _clean(df)
    df = _encode_label(df)

    # Preserve raw education_num integer (used by data_preparation.py for non-IID splitting)
    # before z-score normalization transforms it to floats.
    df["_edu_num_raw"] = df["education_num"].astype(int)

    # Normalize numeric columns — fit on training data
    stats: dict = {}
    for col in NUMERIC_COLS:
        mean = float(df[col].mean())
        std = float(df[col].std())
        df[col] = (df[col] - mean) / (std + 1e-8)
        stats[col] = {"mean": mean, "std": std}

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Save feature column list (everything except label and the raw split key)
    feature_cols = [c for c in df.columns if c not in ("label", "_edu_num_raw")]
    stats["feature_columns"] = feature_cols
    stats["input_dim"] = len(feature_cols)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if stats_output_path:
        Path(stats_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_output_path, "w") as f:
            json.dump(stats, f, indent=2)

    label_dist = df["label"].value_counts().to_dict()
    print(f"[preprocess] {len(df)} rows | {len(feature_cols)} features | labels: {label_dist}")
    print(f"[preprocess] Saved → {output_path}")
    if stats_output_path:
        print(f"[preprocess] Stats  → {stats_output_path}")

    return df, stats


def preprocess_test(input_path: str, output_path: str, stats_path: str):
    """
    Preprocess adult.test using stats fitted on training data.

    adult.test has a descriptor header line and trailing '.' on income labels.
    """
    with open(stats_path) as f:
        stats = json.load(f)

    # adult.test has a descriptor first line — skip it
    df = pd.read_csv(
        input_path, header=None, names=COLUMN_NAMES,
        skipinitialspace=True, skiprows=1,
    )
    df = _clean(df)
    df = _encode_label(df)

    # Apply saved normalization (do NOT refit)
    for col in NUMERIC_COLS:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = (df[col] - mean) / (std + 1e-8)

    # One-hot encode
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Align columns to training feature set (fill missing dummies with 0)
    feature_cols = stats["feature_columns"]
    label = df["label"].copy()
    df = df.drop(columns=["label"], errors="ignore")
    df = df.reindex(columns=feature_cols, fill_value=0)
    df["label"] = label.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    label_dist = df["label"].value_counts().to_dict()
    print(f"[preprocess_test] {len(df)} rows | {len(feature_cols)} features | labels: {label_dist}")
    print(f"[preprocess_test] Saved → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UCI Adult Income dataset")
    parser.add_argument("--input", default="data/adult/adult.data")
    parser.add_argument("--output", default="data/preprocessed/adult_preprocessed.csv")
    parser.add_argument("--stats", default="data/preprocessed/preprocessing_stats.json")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.stats)
