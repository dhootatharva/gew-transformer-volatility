"""LSTM experiment: train-only feature scaling, full-batch Adam on MSE, raw RV target."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from gewvol.config import LSTMConfig
from gewvol.data.sequences import SequenceScaler
from gewvol.models.lstm import LSTMModel
from gewvol.utils import get_device, set_seed

log = logging.getLogger(__name__)


def train_lstm(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, list[dict]]:
    """Full-batch training (one optimiser step per epoch), as in the thesis code."""
    device = get_device()
    model = model.to(device)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        history.append({"epoch": epoch + 1, "train": loss.item(), "val": val_loss})
        log.info("LSTM epoch %d: train=%.6f val=%.6f", epoch + 1, loss.item(), val_loss)
    return model, history


def run_lstm_experiment(
    X: np.ndarray, y: np.ndarray, splits: tuple[slice, slice, slice], cfg: LSTMConfig, seed: int
) -> dict:
    tr, va, te = splits
    scaler = SequenceScaler().fit(X[tr])
    X_train, X_val, X_test = (scaler.transform(X[s]) for s in (tr, va, te))

    set_seed(seed)
    model = LSTMModel(X.shape[-1], cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.head_size)
    model, history = train_lstm(model, X_train, y[tr], X_val, y[va], cfg.epochs, cfg.lr)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32, device=get_device())).cpu().numpy()
    return {"pred": pred, "y_true": y[te], "history": history}
