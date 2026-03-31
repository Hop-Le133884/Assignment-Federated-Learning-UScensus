"""server_app.py — Flower ServerApp for Adult Income federated learning."""

import csv
import torch
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from model import MLP
from task import get_input_dim, load_server_test_data, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Orchestrate federated learning rounds and save metrics to CSV."""
    workspace_path = context.run_config["workspace-path"]
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    batch_size = int(context.run_config["batch-size"])

    # Initialize global model with random weights
    input_dim = get_input_dim(workspace_path)
    global_model = MLP(input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Run federation — evaluate_fn is called after each round
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda rnd, arr: _global_evaluate(rnd, arr, workspace_path, batch_size),
    )

    # Save final model
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print("\nSaved final model → final_model.pt")

    # Save per-round metrics to report/
    report_dir = Path(workspace_path).resolve() / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_metrics(report_dir, result)
    print(f"Reports saved → {report_dir}/")


def _global_evaluate(server_round: int, arrays: ArrayRecord, workspace_path: str, batch_size: int) -> MetricRecord:
    """Evaluate the global model on the centralized server test set."""
    input_dim = get_input_dim(workspace_path)
    model = MLP(input_dim)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    testloader = load_server_test_data(workspace_path, batch_size)
    test_loss, test_acc = test(model, testloader, device)
    print(f"  [Round {server_round}] Global accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def _save_metrics(report_dir: Path, result) -> None:
    """Write accuracy and loss CSVs.

    accuracy_vs_rounds.csv:
      round | global_accuracy (server test set) | communication_round_eval_acc (next round client val)

    loss_vs_rounds.csv:
      round | global_eval_loss (server test set) | communication_round_train_loss (next round client train)
    """
    server_metrics = result.evaluate_metrics_serverapp   # dict[round → MetricRecord]
    client_eval    = result.evaluate_metrics_clientapp   # dict[round → MetricRecord]
    client_train   = result.train_metrics_clientapp      # dict[round → MetricRecord]
    all_rounds     = sorted(server_metrics.keys())       # includes round 0 (pre-training)

    # accuracy_vs_rounds.csv
    # global_accuracy  = server evaluates global model on centralized test set each round
    # communication_round_eval_acc = client eval from the NEXT round (shifted +1)
    with open(report_dir / "accuracy_vs_rounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_accuracy", "communication_round_eval_acc"])
        for rnd in all_rounds:
            global_acc = server_metrics[rnd]["accuracy"]
            client_acc = client_eval[rnd + 1]["eval_acc"] if (rnd + 1) in client_eval else ""
            writer.writerow([rnd, global_acc, client_acc])
    print("  Saved accuracy_vs_rounds.csv")

    # loss_vs_rounds.csv
    # global_eval_loss             = server eval loss of model after round N
    # communication_round_train_loss = client train loss from the NEXT round (shifted +1)
    with open(report_dir / "loss_vs_rounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_eval_loss", "communication_round_train_loss"])
        for rnd in all_rounds:
            eval_loss  = server_metrics[rnd]["loss"]
            train_loss = client_train[rnd + 1]["train_loss"] if (rnd + 1) in client_train else ""
            writer.writerow([rnd, eval_loss, train_loss])
    print("  Saved loss_vs_rounds.csv")
