"""Diebold-Mariano test of equal predictive accuracy under squared-error loss."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


def dm_test(
    y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, hac_lags: int = 0
) -> tuple[float, float]:
    """Return (DM statistic, two-sided p-value).

    d_t = (y - pred1)^2 - (y - pred2)^2. A negative statistic means model 1 has the
    lower squared error. With ``hac_lags=0`` the plain sample variance of d_t is used
    (as in the thesis); otherwise a Newey-West (Bartlett kernel) long-run variance with
    that many lags, which accounts for autocorrelation in d_t. Student-t reference
    with n-1 degrees of freedom.
    """
    y, pred1, pred2 = (np.asarray(a, dtype=float) for a in (y, pred1, pred2))
    if not (len(y) == len(pred1) == len(pred2)):
        raise ValueError("y, pred1 and pred2 must have the same length")
    d = (y - pred1) ** 2 - (y - pred2) ** 2
    n = len(d)
    var = np.var(d)
    dc = d - d.mean()
    for k in range(1, hac_lags + 1):
        var += 2 * (1 - k / (hac_lags + 1)) * np.dot(dc[k:], dc[:-k]) / n
    stat = float(np.mean(d) / np.sqrt(var / n))
    p_value = float(2 * student_t.sf(abs(stat), df=n - 1))
    return stat, p_value
