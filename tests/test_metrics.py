import numpy as np
import pandas as pd
import pytest

from statarb.metrics import (
    annualized_return,
    beta_alpha,
    max_drawdown,
    performance_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.20, 0.05, -0.10, 0.50])
    # equity 1.1, 0.88, 0.924, 0.8316, 1.2474 -> trough 0.8316 vs peak 1.1
    assert max_drawdown(r) == pytest.approx(0.8316 / 1.1 - 1)
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_annualized_return_compounds():
    r = pd.Series([0.01] * 252)
    assert annualized_return(r) == pytest.approx(1.01**252 - 1)


def test_sharpe_scaling_and_degenerate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))


def test_beta_alpha_recovery():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 3000))
    r = 0.0002 + 0.5 * b + pd.Series(rng.normal(0, 0.001, 3000))
    beta, alpha = beta_alpha(r, b)
    assert beta == pytest.approx(0.5, abs=0.01)
    assert alpha == pytest.approx(0.0002 * 252, abs=0.01)


def test_psr_monotone():
    rng = np.random.default_rng(2)
    good = pd.Series(rng.normal(0.001, 0.01, 1000))
    bad = pd.Series(rng.normal(-0.001, 0.01, 1000))
    assert probabilistic_sharpe_ratio(good) > 0.95 > 0.05 > probabilistic_sharpe_ratio(bad)


def test_summary_has_benchmark_fields():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300), idx)
    s = performance_summary(r, benchmark=r)
    assert s["beta"] == pytest.approx(1.0)
    assert s["corr_benchmark"] == pytest.approx(1.0)
