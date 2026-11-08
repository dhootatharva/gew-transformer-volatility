"""Sliding-window datasets for the multi-market GEW and pure-feature models.

Sample t uses feature rows t-L .. t-1 (L = look-back) and targets the
realized volatility of the target market on day t+1, exactly as in the thesis code.
Per market and per day the feature vector is

* GEW  (5 per market): [r, RV, sigma_GARCH, sigma_EGARCH, sigma_EWMA]
* pure (2 per market): [r, RV]

Both datasets are built on the same sample range (the first day with a GEW
feature available), so they share train/val/test targets one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceData:
    X: np.ndarray                 # (samples, lookback, features)
    y: np.ndarray                 # (samples,) target-market RV at t+1
    dates: pd.DatetimeIndex       # date of each target (t+1)
    t_index: np.ndarray           # forecast origin t of each sample (row index into returns)
    feature_names: list[str]


def _sample_range(T: int, burn_in: int, lookback: int) -> range:
    return range(burn_in + lookback, T - 1)


def build_gew_sequences(
    returns: pd.DataFrame,
    rv: pd.DataFrame,
    sigma: dict[str, np.ndarray],
    target: str,
    lookback: int,
    burn_in: int,
) -> SequenceData:
    """Stack [r, RV, GARCH, EGARCH, EWMA] for every market into (samples, lookback, 5*M)."""
    markets = list(returns.columns)
    T = len(returns)
    per_day = []
    for m in markets:
        g, e, w = sigma[m]
        assert len(g) == T - burn_in, "sigma series does not match the return series"
        pad = np.full(burn_in, np.nan)
        per_day.append(np.column_stack([
            returns[m].to_numpy(), rv[m].to_numpy(),
            np.concatenate([pad, g]), np.concatenate([pad, e]), np.concatenate([pad, w]),
        ]))
    day_feats = np.concatenate(per_day, axis=1)  # (T, 5*M)
    names = [f"{m}_{f}" for m in markets for f in ("ret", "rv", "garch", "egarch", "ewma")]
    return _windows(day_feats, rv, target, lookback, _sample_range(T, burn_in, lookback), names)


def build_pure_sequences(
    returns: pd.DataFrame, rv: pd.DataFrame, target: str, lookback: int, burn_in: int
) -> SequenceData:
    """Stack [r, RV] for every market into (samples, lookback, 2*M)."""
    markets = list(returns.columns)
    T = len(returns)
    day_feats = np.concatenate(
        [np.column_stack([returns[m].to_numpy(), rv[m].to_numpy()]) for m in markets], axis=1
    )
    names = [f"{m}_{f}" for m in markets for f in ("ret", "rv")]
    return _windows(day_feats, rv, target, lookback, _sample_range(T, burn_in, lookback), names)


def _windows(
    day_feats: np.ndarray, rv: pd.DataFrame, target: str, lookback: int, ts: range, names: list[str]
) -> SequenceData:
    v = rv[target].to_numpy()
    X = np.stack([day_feats[t - lookback:t] for t in ts])
    y = np.array([v[t + 1] for t in ts])
    dates = pd.DatetimeIndex([rv.index[t + 1] for t in ts])
    return SequenceData(X=X, y=y, dates=dates, t_index=np.array(ts), feature_names=names)


def chrono_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[slice, slice, slice]:
    """Chronological train / validation / test slices (no shuffling)."""
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def clean_and_normalize(
    X: np.ndarray, fit_end: int | None, clip: float = 5.0, gew_variant: bool = True
) -> np.ndarray:
    """Replace NaN/inf, clip, and z-score each feature.

    Statistics are computed on ``X[:fit_end]`` (the training samples). ``fit_end=None``
    uses the full sample, which is what the original notebook did and leaks test data.
    The GEW variant guards tiny standard deviations and applies a final clip; the pure
    variant adds 1e-8 to the std. Both mirror the two original functions.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -clip, clip)
    ref = X if fit_end is None else X[:fit_end]
    mean = ref.mean(axis=(0, 1), keepdims=True)
    std = ref.std(axis=(0, 1), keepdims=True)
    if gew_variant:
        std = np.where(std < 1e-6, 1.0, std)
        return np.clip((X - mean) / std, -clip, clip)
    return (X - mean) / (std + 1e-8)


class SequenceScaler:
    """StandardScaler over the feature axis of a 3-D tensor, fitted on training data only."""

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_
