"""Expanding-window GARCH / EGARCH / EWMA volatility series (the "GEW" features).

For every day t >= burn_in the three models are re-fitted on returns r[0..t-1]
(clipped to +-clip) and their one-step-ahead forecast for day t is stored.
Entry j of each series therefore belongs to day t = burn_in + j and uses only
information available before day t.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from gewvol.models.econometric import egarch_forecast, ewma_forecast, garch_forecast

log = logging.getLogger(__name__)


def compute_sigma_series(
    returns: np.ndarray, burn_in: int = 50, clip: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (garch, egarch, ewma) arrays of length ``len(returns) - burn_in``."""
    r = np.asarray(returns, dtype=float).ravel()
    garch, egarch, ewma = [], [], []
    for t in range(burn_in, len(r)):
        past = np.clip(r[:t], -clip, clip)
        garch.append(garch_forecast(past))
        egarch.append(egarch_forecast(past))
        ewma.append(ewma_forecast(past))
    return np.array(garch), np.array(egarch), np.array(ewma)


def _cache_path(cache_dir: Path, name: str, r: np.ndarray, burn_in: int, clip: float) -> Path:
    digest = hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest()[:12]
    return cache_dir / f"sigma_{name}_{len(r)}_{burn_in}_{clip}_{digest}.npz"


def _worker(args: tuple[str, np.ndarray, int, float, str]) -> tuple[str, np.ndarray]:
    name, r, burn_in, clip, path = args
    g, e, w = compute_sigma_series(r, burn_in, clip)
    out = np.stack([g, e, w])
    np.savez_compressed(path, sigma=out)
    return name, out


def sigma_for_markets(
    returns: pd.DataFrame, burn_in: int, clip: float, cache_dir: Path, n_jobs: int | None = None
) -> dict[str, np.ndarray]:
    """Compute (or load from cache) a (3, T - burn_in) sigma array per market, in parallel."""
    cache_dir = Path(cache_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    todo = []
    for name in returns.columns:
        r = returns[name].to_numpy(dtype=float)
        path = _cache_path(cache_dir, name, r, burn_in, clip)
        if path.exists():
            out[name] = np.load(path)["sigma"]
        else:
            todo.append((name, r, burn_in, clip, str(path)))
    if todo:
        log.info("Fitting GARCH/EGARCH/EWMA for %s (expanding window, this takes a while)",
                 [t[0] for t in todo])
        with ProcessPoolExecutor(max_workers=n_jobs or len(todo)) as pool:
            for name, arr in pool.map(_worker, todo):
                out[name] = arr
    return {name: out[name] for name in returns.columns}
