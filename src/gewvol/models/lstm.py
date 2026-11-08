"""Two-layer LSTM regressor."""

from __future__ import annotations

import torch
from torch import nn


class LSTMModel(nn.Module):
    """Stacked LSTM; the last hidden state goes through a Linear-ReLU-Linear head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        head_size: int = 32,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(nn.Linear(hidden_size, head_size), nn.ReLU(), nn.Linear(head_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, features) -> (batch,)"""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
