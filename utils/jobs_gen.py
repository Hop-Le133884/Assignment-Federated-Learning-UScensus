"""utils/jobs_gen.py — Generate Flower app files into workspace/."""

import argparse
from pathlib import Path

# Generated file templates

MODEL_PY = '''\
"""model.py — MLP for Adult Income binary classification."""

import torch.nn as nn


class MLP(nn.Module):
    """Simple two layers for binary classification.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (batch,) — raw logit
'''

TASK_PY = '''\
"""task.py — Data loading, training, and evaluation functions."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _client_folders(workspace_path: str) -> list:
    """Return sorted list of client_N/ directories found in workspace."""
    workspace = Path(workspace_path).resolve()
    return sorted(p for p in workspace.iterdir() if p.is_dir() and p.name.startswith("client_"))


def get_input_dim(workspace_path: str) -> int:
    """Read feature count from the first available client folder."""
    first = _client_folders(workspace_path)[0]
    df = pd.read_csv(first / "train.csv", nrows=1)
    return len([c for c in df.columns if c != "label"])


def load_data(partition_id: int, workspace_path: str, batch_size: int):
    """Load training data for a client (identified by 0-indexed partition_id).

    Splits the client\'s CSV 80/20 into train and validation sets.
    Returns (trainloader, valloader).
    """
    folders = _client_folders(workspace_path)
    client_dir = folders[partition_id]
    df = pd.read_csv(client_dir / "train.csv")

    feature_cols = [c for c in df.columns if c != "label"]
    X = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32)

    # Deterministic 80/20 split
    n = len(X)
    split = int(0.8 * n)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=g)
    train_idx, val_idx = idx[:split], idx[split:]

    trainloader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    valloader = DataLoader(
        TensorDataset(X[val_idx], y[val_idx]),
        batch_size=batch_size,
    )
    return trainloader, valloader


def load_server_test_data(workspace_path: str, batch_size: int) -> DataLoader:
    """Load the server-side centralized test set."""
    workspace = Path(workspace_path).resolve()
    df = pd.read_csv(workspace / "server" / "test.csv")
    feature_cols = [c for c in df.columns if c != "label"]
    X = torch.tensor(df[feature_cols].astype(float).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


def train(model, trainloader: DataLoader, epochs: int, lr: float, device) -> float:
    """Train model for `epochs` epochs. Returns average training loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()

    running_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / (epochs * len(trainloader))


def test(model, testloader: DataLoader, device) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / len(testloader), correct / total
'''

CLIENT_APP_PY = '''\
"""client_app.py — Flower ClientApp for Adult Income federated learning."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from model import MLP
from task import get_input_dim, load_data
from task import test as test_fn
from task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Receive global model, train on local data, return updated weights."""
    workspace_path = context.run_config["workspace-path"]
    partition_id = context.node_config["partition-id"]
    batch_size = int(context.run_config["batch-size"])

    # Build model and load server weights
    input_dim = get_input_dim(workspace_path)
    model = MLP(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this client\'s local training data
    trainloader, _ = load_data(partition_id, workspace_path, batch_size)

    # Train locally
    train_loss = train_fn(
        model,
        trainloader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(context.run_config["learning-rate"]),
        device=device,
    )

    # Package updated weights and metrics to return to server
    model_record = ArrayRecord(model.state_dict())
    metric_record = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    })
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Receive global model, evaluate on local validation data, return metrics."""
    workspace_path = context.run_config["workspace-path"]
    partition_id = context.node_config["partition-id"]
    batch_size = int(context.run_config["batch-size"])

    # Build model and load server weights
    input_dim = get_input_dim(workspace_path)
    model = MLP(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load this client\'s local validation data
    _, valloader = load_data(partition_id, workspace_path, batch_size)

    # Evaluate
    eval_loss, eval_acc = test_fn(model, valloader, device)

    metric_record = MetricRecord({
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    })
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
'''

SERVER_APP_PY = '''\
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
    print("\\nSaved final model → final_model.pt")

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
    # global_accuracy             = server evaluates global model on centralized test set each round
    # communication_round_eval_acc = client eval on clients's test set and Fedavg them
    with open(report_dir / "accuracy_vs_rounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_accuracy", "communication_round_eval_acc"])
        for rnd in all_rounds:
            global_acc = server_metrics[rnd]["accuracy"]
            client_acc = client_eval[rnd]["eval_acc"] if rnd in client_eval else ""
            writer.writerow([rnd, global_acc, client_acc])
    print("  Saved accuracy_vs_rounds.csv")

    # loss_vs_rounds.csv
    with open(report_dir / "loss_vs_rounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "global_eval_loss", "communication_round_train_loss"])
        for rnd in all_rounds:
            eval_loss  = server_metrics[rnd]["loss"]
            train_loss = client_train[rnd]["train_loss"] if rnd in client_train else ""
            writer.writerow([rnd, eval_loss, train_loss])
    print("  Saved loss_vs_rounds.csv")
'''

PYPROJECT_TOML = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "adult-federated"
version = "1.0.0"
description = "Federated Learning on UCI Adult Income with Flower"
license = "Apache-2.0"
dependencies = [
    "flwr[simulation]>=1.26.0",
    "torch==2.8.0",
    "pandas>=3.0.0",
    "numpy>=2.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "local"

[tool.flwr.app.components]
serverapp = "server_app:app"
clientapp = "client_app:app"

[tool.flwr.app.config]
num-server-rounds = 10
fraction-evaluate = 0.6
local-epochs = 1
learning-rate = 0.01
batch-size = 64
workspace-path = "./workspace"
'''

GITIGNORE = '''\
# Virtual environment — excluded from Flower FAB bundle
.venv/

# Data files — excluded from Flower FAB bundle (too large)
*.csv
*.pt
*.pth

# Python cache
__pycache__/
*.pyc
*.pyo
'''


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  wrote {path}")


def generate(workspace: str, num_clients: int):
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating Flower app files in {ws}/...")

    # Shared workspace-root files (the actual Flower app)
    _write(ws / "model.py", MODEL_PY)
    _write(ws / "task.py", TASK_PY)
    _write(ws / "client_app.py", CLIENT_APP_PY)
    _write(ws / "server_app.py", SERVER_APP_PY)
    _write(ws / "pyproject.toml", PYPROJECT_TOML)
    _write(ws / ".gitignore", GITIGNORE)

    print(f"\nDone. Run 'docker..' to start training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Flower workspace app files")
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--num-clients", type=int, default=5)
    args = parser.parse_args()
    generate(args.workspace, args.num_clients)
