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

    # Load this client's local training data
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

    # Load this client's local validation data
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
