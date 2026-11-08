"""Log returns and rolling realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_realized_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised rolling realized volatility: sample std of the last ``window`` returns x sqrt(252).

    The value at date t uses returns t-window+1 .. t (trailing window, no future data).
    """
    return (returns.rolling(window).std() * np.sqrt(252)).dropna()


def compute_features(prices: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns, rv) on a common date index."""
    returns = compute_log_returns(prices)
    rv = compute_realized_volatility(returns, window)
    return returns.loc[rv.index], rv
