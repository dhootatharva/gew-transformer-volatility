"""Volatility forecast loss functions (Patton, 2011)."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def hmae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Heteroskedasticity-adjusted MAE: mean |1 - y_hat / y|."""
    return float(np.mean(np.abs((y - y_hat) / (y + EPS))))


def hmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Heteroskedasticity-adjusted MSE: mean (1 - y_hat / y)^2."""
    return float(np.mean(((y - y_hat) / (y + EPS)) ** 2))


def qlike(y: np.ndarray, y_hat: np.ndarray) -> float:
    """QLIKE: mean(y/y_hat - ln(y/y_hat) - 1). Non-negative, zero iff y_hat == y."""
    ratio = y / (y_hat + EPS)
    return float(np.mean(ratio - np.log(ratio) - 1))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / (ss_tot + EPS))


def evaluate_all(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    return {
        "RMSE": rmse(y, y_hat),
        "MAE": mae(y, y_hat),
        "HMAE": hmae(y, y_hat),
        "HMSE": hmse(y, y_hat),
        "QLIKE": qlike(y, y_hat),
        "R2": r2(y, y_hat),
    }
