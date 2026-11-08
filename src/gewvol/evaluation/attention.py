"""Turn the averaged (T*M, T*M) token attention matrix into lag and market summaries.

Tokens are ordered time-major: token index = time * M + market.
"""

from __future__ import annotations

import numpy as np


def lag_importance(attn: np.ndarray, num_markets: int) -> np.ndarray:
    """Mean attention received by each time step (length T), markets collapsed."""
    T = attn.shape[0] // num_markets
    per_time = attn.reshape(T, num_markets, T, num_markets).mean(axis=(1, 3))
    return per_time.mean(axis=0)


def market_spillover(attn: np.ndarray, num_markets: int) -> np.ndarray:
    """(M, M) matrix: row = querying market, column = attended market; rows sum to 1."""
    N = attn.shape[0]
    if N % num_markets:
        raise ValueError("attention matrix size is not a multiple of num_markets")
    T = N // num_markets
    spill = attn.reshape(T, num_markets, T, num_markets).sum(axis=(0, 2))
    return spill / (spill.sum(axis=1, keepdims=True) + 1e-8)
