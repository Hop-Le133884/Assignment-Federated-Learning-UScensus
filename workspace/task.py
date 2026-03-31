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

    Splits the client's CSV 80/20 into train and validation sets.
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
