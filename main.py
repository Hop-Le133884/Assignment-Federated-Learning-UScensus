"""
Entry point for the Federated Learning experiment on the US Census dataset.

Run with:
    uv run python main.py
or:
    python main.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from utils.prepare_data_splits import load_adult_both_splits, preprocess_and_split
from jobs.config.fl_simulation import run_federated_learning, run_centralized_baseline


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main() -> None:
    print_section("Federated Learning – US Census Income Dataset")

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading dataset …")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_adult_both_splits()
    print(f"      Raw training rows : {len(X_train_raw):,}")
    print(f"      Raw test rows     : {len(X_test_raw):,}")

    # ------------------------------------------------------------------
    # 2. Preprocess + create non-IID client splits
    # ------------------------------------------------------------------
    print("\n[2/4] Preprocessing and creating non-IID client splits …")
    (X_train, y_train, X_test, y_test,
     client_data, feature_names, client_info) = preprocess_and_split(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw
    )

    print(f"\n      Training samples : {X_train.shape[0]:,}")
    print(f"      Test samples     : {X_test.shape[0]:,}")
    print(f"      Features         : {X_train.shape[1]}")
    print(f"\n      Non-IID distribution (partitioned by education level):")
    for cid, info in client_info.items():
        label = info['label'].replace('\n', ' ')
        print(
            f"        Client {cid}  [{label:35s}]  "
            f"{info['n_samples']:5,} samples   "
            f"{info['high_income_pct']:5.1f}% earn >50K"
        )

    # ------------------------------------------------------------------
    # 3. Run Federated Learning
    # ------------------------------------------------------------------
    print_section("Federated Learning (FedAvg) – 20 rounds, 5 clients")
    fl_results = run_federated_learning(
        client_data=client_data,
        X_test=X_test,
        y_test=y_test,
        n_rounds=20,
        local_epochs=5,
        learning_rate=0.01,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 4. Centralised baseline
    # ------------------------------------------------------------------
    print_section("Centralised baseline (100 epochs, full training set)")
    central = run_centralized_baseline(
        X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01
    )
    print(f"\n  Loss     : {central['loss']:.4f}")
    print(f"  Accuracy : {central['accuracy']:.4f}  ({central['accuracy'] * 100:.1f}%)")

    final_acc  = fl_results['accuracies'][-1]
    final_loss = fl_results['losses'][-1]
    print(f"\n  FL final (round 20): Loss={final_loss:.4f}, Accuracy={final_acc:.4f} ({final_acc*100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Required graphs
    # ------------------------------------------------------------------
    rounds = list(range(1, fl_results['n_rounds'] + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Federated Learning – US Census Income Dataset (5 clients, non-IID by education)',
        fontsize=13, fontweight='bold'
    )

    # Graph 1 – Accuracy vs. Communication Rounds
    ax1.plot(rounds, fl_results['accuracies'], 'b-o',
             linewidth=2, markersize=5, label='FL global model')
    ax1.axhline(central['accuracy'], color='red', linestyle='--', linewidth=1.5,
                label=f"Centralised ({central['accuracy']:.3f})")
    ax1.set_xlabel('Communication Round', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Graph 1: Global Model Accuracy\nvs. Communication Rounds', fontsize=11)
    ax1.set_xlim(1, fl_results['n_rounds'])
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Graph 2 – Loss vs. Communication Rounds
    ax2.plot(rounds, fl_results['losses'], 'r-o',
             linewidth=2, markersize=5, label='FL global model')
    ax2.axhline(central['loss'], color='blue', linestyle='--', linewidth=1.5,
                label=f"Centralised ({central['loss']:.3f})")
    ax2.set_xlabel('Communication Round', fontsize=11)
    ax2.set_ylabel('Loss (Binary Cross-Entropy)', fontsize=11)
    ax2.set_title('Graph 2: Global Model Loss\nvs. Communication Rounds', fontsize=11)
    ax2.set_xlim(1, fl_results['n_rounds'])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fl_results.png', dpi=150, bbox_inches='tight')
    print("\n  Graphs saved → fl_results.png")
    plt.show()


if __name__ == "__main__":
    main()
