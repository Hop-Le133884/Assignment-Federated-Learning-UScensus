"""model.py — MLP for Adult Income binary classification."""

import torch.nn as nn


class MLP(nn.Module):
    """simple NN 2 learning layers.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
