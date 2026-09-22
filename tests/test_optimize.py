import numpy as np
import pytest

from statarb.covariance import ewma_cov, ledoit_wolf, sample_cov
from statarb.optimize import (
    inverse_volatility,
    max_sharpe,
    min_variance,
    risk_contributions,
    risk_parity,
)


def _cov(seed=0, n=5):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 0.05


def _valid(w, cap=1.0):
    assert w.sum() == pytest.approx(1.0)
    assert w.min() >= -1e-12 and w.max() <= cap + 1e-9


def test_min_variance_diagonal_is_inverse_variance():
    var = np.array([0.04, 0.09, 0.01])
    w = min_variance(np.diag(var))
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum(), atol=1e-6)


def test_min_variance_beats_equal_weight_and_respects_cap():
    cov = _cov()
    w = min_variance(cov, max_weight=0.3)
    _valid(w, 0.3)
    ew = np.full(5, 0.2)
    assert w @ cov @ w <= ew @ cov @ ew + 1e-12


def test_max_sharpe_matches_closed_form_when_unconstrained():
    cov = np.diag([0.04, 0.09])
    mu = np.array([0.08, 0.09])
    w = max_sharpe(mu, cov)
    raw = np.linalg.solve(cov, mu)
    np.testing.assert_allclose(w, raw / raw.sum(), atol=1e-5)


def test_max_sharpe_cap_and_fallback():
    cov = _cov(1)
    mu = np.array([0.2, 0.01, 0.01, 0.01, 0.01])
    w = max_sharpe(mu, cov, max_weight=0.4)
    _valid(w, 0.4)
    with pytest.warns(UserWarning):
        w2 = max_sharpe(-np.ones(5) * 0.01, cov)
    np.testing.assert_allclose(w2, min_variance(cov), atol=1e-6)
    with pytest.raises(ValueError):
        max_sharpe(mu, cov, max_weight=0.1)  # 5 * 0.1 < 1 infeasible


def test_risk_parity_equalises_contributions():
    cov = _cov(2, n=6)
    w = risk_parity(cov)
    _valid(w)
    np.testing.assert_allclose(risk_contributions(w, cov), np.full(6, 1 / 6), atol=1e-6)


def test_risk_parity_budgets_and_uncorrelated_case():
    var = np.array([0.01, 0.04, 0.16])
    np.testing.assert_allclose(
        risk_parity(np.diag(var)), inverse_volatility(np.diag(var)), atol=1e-6
    )
    b = np.array([0.5, 0.3, 0.2])
    cov = _cov(3, n=3)
    np.testing.assert_allclose(risk_contributions(risk_parity(cov, b), cov), b, atol=1e-6)


def test_ledoit_wolf_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.covariance")
    X = np.random.default_rng(4).normal(size=(60, 8)) @ np.diag(np.linspace(0.5, 2, 8))
    cov, s = ledoit_wolf(X)
    ref = sklearn.LedoitWolf().fit(X)
    np.testing.assert_allclose(cov, ref.covariance_, rtol=1e-10, atol=1e-12)
    assert s == pytest.approx(ref.shrinkage_)


def test_ledoit_wolf_properties():
    X = np.random.default_rng(5).normal(size=(30, 20))
    cov, s = ledoit_wolf(X)
    assert 0 <= s <= 1
    assert np.linalg.eigvalsh(cov).min() > 0  # well-conditioned even when T ~ N


def test_ewma_reduces_to_sample_for_huge_halflife():
    X = np.random.default_rng(6).normal(size=(200, 3))
    np.testing.assert_allclose(ewma_cov(X, 1e9), sample_cov(X), rtol=1e-6)
