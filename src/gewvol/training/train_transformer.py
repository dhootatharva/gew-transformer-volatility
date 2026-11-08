"""Transformer experiment: standardised target, mini-batch Adam on Huber loss."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from gewvol.config import TransformerConfig
from gewvol.models.transformer import TransformerModel
from gewvol.utils import get_device, set_seed

log = logging.getLogger(__name__)


def train_transformer(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TransformerConfig,
) -> tuple[nn.Module, list[dict]]:
    device = get_device()
    model = model.to(device)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(Xv)
            val_loss = criterion(val_pred, yv).item()
        history.append({"epoch": epoch + 1, "train": total / len(loader), "val": val_loss})
        log.info("Transformer epoch %d: train=%.6f val=%.6f", epoch + 1, total / len(loader), val_loss)
    return model, history


@torch.no_grad()
def predict_with_attention(
    model: TransformerModel, X: np.ndarray, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Predict in batches and return (predictions, attention averaged over layers, heads and samples)."""
    device = get_device()
    model.eval()
    preds, attn_sum, n = [], None, 0
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        p, attn = model(xb)
        preds.append(p.cpu().numpy())
        a = torch.stack(attn).mean(dim=0).sum(dim=0).cpu().numpy()  # mean layers, sum batch
        attn_sum = a if attn_sum is None else attn_sum + a
        n += len(xb)
    return np.concatenate(preds), attn_sum / n


def run_transformer_experiment(
    X: np.ndarray,
    y: np.ndarray,
    splits: tuple[slice, slice, slice],
    cfg: TransformerConfig,
    seed: int,
    num_markets: int,
    target_stats_on_train: bool = True,
) -> dict:
    """Train on standardised y and return de-standardised test predictions plus attention.

    ``target_stats_on_train=False`` reproduces the original notebook, which standardised
    y with the mean/std of the full sample (including the test period).
    """
    set_seed(seed)
    tr, va, te = splits
    ref = y[tr] if target_stats_on_train else y
    y_mean, y_std = float(ref.mean()), float(ref.std()) + 1e-8
    y_norm = (y - y_mean) / y_std

    model = TransformerModel(
        input_size=X.shape[-1], num_markets=num_markets, d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_layers=cfg.num_layers, dim_ff=cfg.dim_ff, dropout=cfg.dropout,
        head_size=cfg.head_size, max_len=cfg.max_len,
    )
    model, history = train_transformer(model, X[tr], y_norm[tr], X[va], y_norm[va], cfg)
    pred, attn = predict_with_attention(model, X[te])
    return {"pred": pred * y_std + y_mean, "y_true": y[te], "attn": attn, "history": history}
