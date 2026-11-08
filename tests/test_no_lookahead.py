"""Checks that no feature, statistic or forecast uses information from the future."""

import numpy as np
import pandas as pd
import pytest

from gewvol.data.features import compute_features
from gewvol.data.sequences import (
    build_gew_sequences,
    build_pure_sequences,
    chrono_split,
    clean_and_normalize,
)
from gewvol.data.sigma import compute_sigma_series

MARKETS = ["NIFTY", "SP500", "KOSPI"]
rng = np.random.default_rng(42)


def _prices(n=260):
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = rng.normal(0, 0.01, size=(n, 3))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=MARKETS)


def _fake_sigma(returns, burn_in):
    T = len(returns)
    return {m: np.abs(rng.normal(0.01, 0.002, size=(3, T - burn_in))) for m in returns.columns}


def test_realized_volatility_is_trailing():
    prices = _prices()
    r1, v1 = compute_features(prices, window=20)
    shocked = prices.copy()
    shocked.iloc[150:] *= 1.5
    _, v2 = compute_features(shocked, window=20)
    cut = prices.index[148]
    pd.testing.assert_frame_equal(v1.loc[:cut], v2.loc[:cut])


def test_sigma_series_uses_only_past_returns():
    r = rng.normal(0, 0.01, size=36)
    burn_in = 30
    base = compute_sigma_series(r, burn_in=burn_in)
    r2 = r.copy()
    r2[33:] = 0.08  # change returns from day 33 onwards
    moved = compute_sigma_series(r2, burn_in=burn_in)
    # entries for days 30..33 use returns up to day t-1 <= 32 and must be identical
    for a, b in zip(base, moved):
        np.testing.assert_allclose(a[:4], b[:4])
        assert not np.allclose(a[4:], b[4:])


@pytest.mark.parametrize("kind", ["gew", "pure"])
def test_sequence_window_ends_before_origin_and_target_is_t_plus_1(kind):
    prices = _prices()
    returns, rv = compute_features(prices, window=20)
    burn_in, lookback = 30, 25
    sigma = _fake_sigma(returns, burn_in)

    def build(ret, vol):
        if kind == "gew":
            return build_gew_sequences(ret, vol, sigma, "NIFTY", lookback, burn_in)
        return build_pure_sequences(ret, vol, "NIFTY", lookback, burn_in)

    ds = build(returns, rv)
    i = 40
    t = ds.t_index[i]
    assert ds.y[i] == rv["NIFTY"].iloc[t + 1]
    assert ds.dates[i] == rv.index[t + 1]

    ret2, rv2 = returns.copy(), rv.copy()
    ret2.iloc[t:] = 0.5
    rv2.iloc[t:] = 9.0
    ds2 = build(ret2, rv2)
    np.testing.assert_array_equal(ds.X[i], ds2.X[i])


def test_gew_and_pure_share_samples():
    prices = _prices()
    returns, rv = compute_features(prices, window=20)
    gew = build_gew_sequences(returns, rv, _fake_sigma(returns, 30), "NIFTY", 25, 30)
    pure = build_pure_sequences(returns, rv, "NIFTY", 25, 30)
    np.testing.assert_array_equal(gew.y, pure.y)
    assert gew.X.shape[2] == 15 and pure.X.shape[2] == 6


def test_chrono_split_is_ordered_and_disjoint():
    tr, va, te = chrono_split(1000, 0.7, 0.15)
    assert (tr.start, tr.stop, va.start, va.stop, te.start, te.stop) == (0, 700, 700, 850, 850, 1000)


@pytest.mark.parametrize("gew_variant", [True, False])
def test_normalisation_statistics_ignore_test_data(gew_variant):
    X = rng.normal(0.2, 0.05, size=(200, 10, 4))
    fit_end = 140
    a = clean_and_normalize(X, fit_end, gew_variant=gew_variant)
    X2 = X.copy()
    X2[fit_end:] *= 3.0
    b = clean_and_normalize(X2, fit_end, gew_variant=gew_variant)
    np.testing.assert_allclose(a[:fit_end], b[:fit_end])
    np.testing.assert_allclose(a[:fit_end].mean(axis=(0, 1)), 0, atol=1e-6)
    # the notebook's global normalisation does depend on test data
    g1 = clean_and_normalize(X, None, gew_variant=gew_variant)
    g2 = clean_and_normalize(X2, None, gew_variant=gew_variant)
    assert not np.allclose(g1[:fit_end], g2[:fit_end])


def test_econometric_baseline_uses_sigma_for_forecast_origin():
    from gewvol.config import Config
    from gewvol.pipeline import econometric_baselines

    cfg = Config()
    burn_in = cfg.data.sigma_burn_in
    T = 400
    t_index = np.arange(burn_in + 60, T - 1)
    g = np.arange(T - burn_in, dtype=float)  # sigma value = day index - burn_in
    sigma = {cfg.data.target_market: np.stack([g, g + 1000, g + 2000])}

    class _DS:
        pass

    gew = _DS()
    gew.t_index = t_index
    splits = chrono_split(len(t_index))
    out = econometric_baselines({"sigma": sigma, "gew": gew, "splits": splits}, cfg)
    expected_day = t_index[splits[2]]  # forecast for day t, fitted on returns up to t-1
    np.testing.assert_allclose(out["GARCH(1,1)"], (expected_day - burn_in) * np.sqrt(252))
