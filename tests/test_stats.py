import numpy as np
import pandas as pd
import pytest

from statarb.stats import (
    benjamini_hochberg,
    engle_granger,
    half_life,
    ols_hedge_ratio,
    rolling_zscore,
)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.3 + 1.7 * x + rng.normal(scale=0.01, size=500)
    a, b = ols_hedge_ratio(y, x)
    assert a == pytest.approx(0.3, abs=0.01)
    assert b == pytest.approx(1.7, abs=0.01)


def test_engle_granger_detects_cointegration(market):
    prices, truth = market
    t = truth[0]
    lp = np.log(prices)
    res = engle_granger(lp[t.y].to_numpy(), lp[t.x].to_numpy())
    assert res.pvalue < 0.01
    assert res.beta == pytest.approx(t.hedge_ratio, rel=0.1)


def test_engle_granger_rejects_independent_walks():
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(40):
        a = np.cumsum(rng.normal(size=300))
        b = np.cumsum(rng.normal(size=300))
        rejections += engle_granger(a, b).pvalue < 0.05
    assert rejections <= 7  # nominal 5% of 40 = 2; generous bound


@pytest.mark.parametrize("true_hl", [3.0, 10.0, 25.0])
def test_half_life_of_ar1(true_hl):
    rng = np.random.default_rng(1)
    phi = 0.5 ** (1 / true_hl)
    s = np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert half_life(s) == pytest.approx(true_hl, rel=0.15)


def test_half_life_edge_cases():
    assert half_life(1.02 ** np.arange(200)) == float("inf")  # explosive, phi > 1
    alt = np.array([(-1) ** i for i in range(100)], dtype=float)
    assert np.isnan(half_life(alt))
    with pytest.raises(ValueError):
        half_life(np.arange(5.0))


def test_benjamini_hochberg_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # thresholds q*k/m with q=0.05, m=10: 0.005, 0.01, 0.015, ...; only first two pass
    np.testing.assert_array_equal(benjamini_hochberg(p, 0.05), [True, True] + [False] * 8)
    # step-up: a later p below its threshold rejects all smaller ones too
    p2 = np.array([0.04, 0.001, 0.03])  # sorted 0.001, 0.03, 0.04 vs 0.0167, 0.0333, 0.05
    np.testing.assert_array_equal(benjamini_hochberg(p2, 0.05), [True, True, True])
    assert benjamini_hochberg(np.array([]), 0.05).size == 0


def test_rolling_zscore_is_causal():
    s = pd.Series(np.random.default_rng(0).normal(size=100))
    z1 = rolling_zscore(s, 20)
    s2 = s.copy()
    s2.iloc[60:] += 100.0
    z2 = rolling_zscore(s2, 20)
    pd.testing.assert_series_equal(z1.iloc[:60], z2.iloc[:60])
    assert z1.iloc[:19].isna().all()
