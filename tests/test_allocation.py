import numpy as np
import pandas as pd
import pytest

from statarb.allocation import allocation_targets, combine_sleeves, run_allocation
from statarb.config import AllocationConfig, CostModel


@pytest.fixture
def alloc_prices(market):
    prices, _ = market
    return prices[[c for c in prices.columns if c.startswith("IDX")]]


@pytest.mark.parametrize(
    "method", ["equal", "inverse_vol", "min_variance", "max_sharpe", "risk_parity"]
)
def test_targets_valid_for_every_method(alloc_prices, method):
    cfg = AllocationConfig(method=method, lookback_days=120, rebalance_every=40, max_weight=0.6)
    t = allocation_targets(alloc_prices, cfg)
    np.testing.assert_allclose(t.sum(axis=1), 1.0, atol=1e-8)
    assert (t >= -1e-12).all().all()
    assert t.index[0] == alloc_prices.index[120]


@pytest.mark.filterwarnings("ignore:no asset beats rf")
def test_allocation_has_no_lookahead(alloc_prices):
    cfg = AllocationConfig(method="max_sharpe", lookback_days=120, rebalance_every=20)
    cut = 500
    base = allocation_targets(alloc_prices, cfg)
    pert = alloc_prices.copy()
    pert.iloc[cut:] = pert.iloc[cut:] * np.linspace(1, 3, len(pert) - cut)[:, None]
    new = allocation_targets(pert, cfg)
    pd.testing.assert_frame_equal(
        base.loc[: alloc_prices.index[cut - 1]], new.loc[: alloc_prices.index[cut - 1]]
    )


def test_trend_filter_goes_to_cash(alloc_prices):
    falling = alloc_prices.copy()
    falling["IDX0"] = 100 * np.exp(-0.001 * np.arange(len(falling)))
    cfg = AllocationConfig(method="equal", lookback_days=60, trend_filter=True, trend_lookback=60)
    t = allocation_targets(falling, cfg)
    assert (t["IDX0"] == 0).all()
    assert (t.sum(axis=1) <= 1 + 1e-12).all()


def test_run_allocation_trimmed(alloc_prices):
    cfg = AllocationConfig(lookback_days=120)
    res = run_allocation(alloc_prices, cfg, CostModel())
    assert res.returns.index[0] == alloc_prices.index[120]
    assert np.isfinite(res.summary()["sharpe"])


def test_combine_sleeves_math():
    idx = pd.bdate_range("2021-01-01", periods=4)
    a = pd.Series([0.0, 0.10, 0.0, 0.0], idx)
    b = pd.Series([0.0, -0.10, 0.0, 0.0], idx)
    res = combine_sleeves({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, rebalance_every=1)
    assert res.returns.iloc[1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        combine_sleeves({"a": a}, {"a": 1.2})
