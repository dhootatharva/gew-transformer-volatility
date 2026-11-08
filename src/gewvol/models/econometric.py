"""GARCH(1,1), EGARCH(1,1) and EWMA (RiskMetrics) one-step-ahead volatility forecasts.

Returns are scaled by 100 for numerical stability of the ``arch`` optimiser and the
forecast is scaled back, so outputs are daily (non-annualised) standard deviations.
"""

from __future__ import annotations

import warnings

import numpy as np
from arch import arch_model


def _fit_forecast(returns: np.ndarray, vol: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(returns * 100, vol=vol, p=1, q=1).fit(disp="off", show_warning=False)
        var = res.forecast(horizon=1).variance.values[-1, 0]
    return float(np.sqrt(var) / 100)


def garch_forecast(returns: np.ndarray) -> float:
    """One-step-ahead sigma from a GARCH(1,1) with constant mean, fitted by MLE."""
    try:
        return _fit_forecast(returns, "Garch")
    except Exception:
        return float("nan")


def egarch_forecast(returns: np.ndarray) -> float:
    """One-step-ahead sigma from an EGARCH(1,1); NaN if the fit fails."""
    try:
        return _fit_forecast(returns, "EGarch")
    except Exception:
        return float("nan")


def ewma_forecast(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA sigma, initialised at the sample variance of ``returns``."""
    var = float(np.var(returns))
    for r in returns:
        var = lam * var + (1 - lam) * r**2
    return float(np.sqrt(var))
