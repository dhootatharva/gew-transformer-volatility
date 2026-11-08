import numpy as np
import pytest

from gewvol.evaluation.stat_tests import dm_test

rng = np.random.default_rng(1)
Y = rng.uniform(0.1, 0.5, size=400)


def test_better_model_has_negative_significant_stat():
    good = Y + rng.normal(0, 0.01, Y.size)
    bad = Y + rng.normal(0, 0.05, Y.size)
    stat, p = dm_test(Y, good, bad)
    assert stat < 0 and p < 1e-6
    stat_rev, p_rev = dm_test(Y, bad, good)
    assert stat_rev == pytest.approx(-stat) and p_rev == pytest.approx(p)


def test_equal_models_not_significant():
    a = Y + rng.normal(0, 0.02, Y.size)
    b = Y + rng.normal(0, 0.02, Y.size)
    _, p = dm_test(Y, a, b)
    assert p > 0.01


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        dm_test(Y, Y[:-1], Y)


def test_newey_west_widens_variance_for_autocorrelated_losses():
    n = 600
    noise = np.cumsum(rng.normal(0, 0.002, n))  # persistent error component
    y = np.full(n, 0.2)
    p1 = y + 0.01 + noise
    p2 = y + 0.012 + noise
    s0, _ = dm_test(y, p1, p2)
    s_nw, _ = dm_test(y, p1, p2, hac_lags=10)
    assert abs(s_nw) < abs(s0)
    assert dm_test(y, p1, p2, hac_lags=0) == (s0, dm_test(y, p1, p2)[1])
