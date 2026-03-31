import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ACC_CSV = "./workspace/report/accuracy_vs_rounds.csv"
LOSS_CSV = "./workspace/report/loss_vs_rounds.csv"

acc_df = pd.read_csv(ACC_CSV)
loss_df = pd.read_csv(LOSS_CSV)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Federated Learning Results", fontsize=14, fontweight="bold")

# --- Bar chart: Accuracy vs Rounds ---
rounds = acc_df["round"]
bar_width = 0.35
x = rounds.to_numpy()

bars1 = ax1.bar(x - bar_width / 2, acc_df["global_accuracy"], bar_width,
                label="Global Accuracy", color="steelblue")
bars2 = ax1.bar(x + bar_width / 2, acc_df["communication_round_eval_acc"], bar_width,
                label="Comm. Round Eval Acc", color="coral")

ax1.set_title("Global Accuracy vs Communication Rounds")
ax1.set_xlabel("Round")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(x)
ax1.set_ylim(0, 1.05)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# --- Line chart: Loss vs Rounds ---
ax2.plot(loss_df["round"], loss_df["global_eval_loss"],
         marker="o", label="Global Eval Loss", color="steelblue")
ax2.plot(loss_df["round"], loss_df["communication_round_train_loss"],
         marker="s", linestyle="--", label="Comm. Round Train Loss", color="coral")

ax2.set_title("Global Loss vs Communication Rounds")
ax2.set_xlabel("Round")
ax2.set_ylabel("Loss")
ax2.set_xticks(loss_df["round"])
ax2.legend()
ax2.grid(linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("./workspace/report/acc_loss_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to /workspace/report/acc_loss_visualization.png")
