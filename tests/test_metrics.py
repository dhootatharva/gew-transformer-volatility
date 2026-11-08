import numpy as np
import pytest

from gewvol.evaluation.metrics import evaluate_all, hmae, hmse, mae, qlike, rmse

rng = np.random.default_rng(0)
Y = rng.uniform(0.05, 0.6, size=500)


def test_perfect_forecast_has_zero_loss():
    m = evaluate_all(Y, Y.copy())
    for k in ("RMSE", "MAE", "HMAE", "HMSE"):
        assert m[k] == pytest.approx(0.0, abs=1e-12)
    assert m["QLIKE"] == pytest.approx(0.0, abs=1e-6)
    assert m["R2"] == pytest.approx(1.0)


def test_qlike_non_negative_and_asymmetric():
    over = qlike(Y, Y * 1.2)
    under = qlike(Y, Y * 0.8)
    assert over >= 0 and under >= 0
    assert under > over  # QLIKE penalises under-prediction more


def test_known_values():
    y = np.array([1.0, 2.0])
    p = np.array([2.0, 2.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(0.5))
    assert mae(y, p) == pytest.approx(0.5)
    assert hmae(y, p) == pytest.approx(0.5, rel=1e-6)
    assert hmse(y, p) == pytest.approx(0.5, rel=1e-6)
    assert qlike(y, p) == pytest.approx(0.5 * (0.5 - np.log(0.5) - 1), rel=1e-6)
