"""utils/data_preparation.py — Split preprocessed data non-IID across clients."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow importing utils from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_preprocessing import preprocess_test

# Non-IID split: education_num values (1–16) grouped into 5 buckets.

EDUCATION_BUCKETS_5 = {
    1: list(range(1, 6)),    # Preschool, 1st-4th, 5th-6th, 7th-8th, 9th
    2: list(range(6, 9)),    # 10th, 11th, 12th
    3: [9],                  # HS-grad
    4: list(range(10, 13)),  # Some-college, Assoc-voc, Assoc-acdm
    5: list(range(13, 17)),  # Bachelors, Masters, Prof-school, Doctorate
}


def _make_buckets(df: pd.DataFrame, num_clients: int) -> dict:
    """Return {client_id: [education_num values]} mapping."""
    if num_clients == 5:
        return EDUCATION_BUCKETS_5

    # else will return bins coresponding with unique raw education_num values
    unique_vals = sorted(df["_edu_num_raw"].unique())
    bins = [[] for _ in range(num_clients)]
    for i, val in enumerate(unique_vals):
        bins[i % num_clients].append(val)
    return {i + 1: bins[i] for i in range(num_clients)}


def split_data(num_clients: int, workspace: str, preprocessed_csv: str, stats_path: str, test_input: str):
    workspace = Path(workspace)

    print(f"\nLoading preprocessed data from {preprocessed_csv}")
    df = pd.read_csv(preprocessed_csv)
    feature_count = df.shape[1] - 2  # subtract label and _edu_num_raw
    print(f"Total rows: {len(df)} | Features: {feature_count} | Label dist: {df['label'].value_counts().to_dict()}")

    buckets = _make_buckets(df, num_clients)

    print(f"\nSplitting into {num_clients} clients (non-IID by education level):")
    print(f"{'Client':<10} {'Edu levels':<30} {'Samples':<10} {'Label=1 %':<12}")
    print("-" * 65)

    for client_id in range(1, num_clients + 1):
        edu_vals = buckets[client_id]
        # Filter by raw integer education_num (preserved during preprocessing as _edu_num_raw)
        client_df = df[df["_edu_num_raw"].isin(edu_vals)].drop(columns=["_edu_num_raw"])
        client_dir = workspace / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        client_df.to_csv(client_dir / "train.csv", index=False)

        label_pct = client_df["label"].mean() * 100
        print(f"  client_{client_id:<4} {str(edu_vals):<30} {len(client_df):<10} {label_pct:.1f}%")

    # Server test set — preprocessed from adult.test with saved stats
    print(f"\nProcessing server test set from {test_input}")
    server_dir = workspace / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    preprocess_test(
        input_path=test_input,
        output_path=str(server_dir / "test.csv"),
        stats_path=stats_path,
    )

    print(f"\nAll splits written to {workspace}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Adult data non-IID across FL clients")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--preprocessed-csv", default="data/preprocessed/adult_preprocessed.csv")
    parser.add_argument("--stats-path", default="data/preprocessed/preprocessing_stats.json")
    parser.add_argument("--test-input", default="data/adult/adult.test")
    args = parser.parse_args()

    split_data(
        num_clients=args.num_clients,
        workspace=args.workspace,
        preprocessed_csv=args.preprocessed_csv,
        stats_path=args.stats_path,
        test_input=args.test_input,
    )
