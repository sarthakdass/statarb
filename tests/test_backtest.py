import numpy as np
import pandas as pd
import pytest

from statarb.backtest import run_backtest, targets_on_change
from statarb.config import CostModel


def _prices():
    idx = pd.bdate_range("2021-01-01", periods=6)
    return pd.DataFrame(
        {"A": [100, 110, 99, 99, 120, 108.0], "B": [50, 50, 55, 44, 44, 46.2]}, index=idx
    )


def test_full_long_matches_asset_returns():
    p = _prices()
    tgt = pd.DataFrame({"A": 1.0}, index=p.index[:1])
    res = run_backtest(p, tgt, CostModel.zero(), execution_lag=0)
    expected = p["A"].pct_change().fillna(0.0)
    np.testing.assert_allclose(res.returns.to_numpy(), expected.to_numpy())
    assert res.equity.iloc[-1] == pytest.approx(1.08)


def test_buy_and_hold_drift_equals_average_of_normalised_prices():
    p = _prices()
    tgt = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=p.index[:1])
    res = run_backtest(p, tgt, CostModel.zero(), execution_lag=0)
    bh = 0.5 * p["A"] / p["A"].iloc[0] + 0.5 * p["B"] / p["B"].iloc[0]
    np.testing.assert_allclose(res.equity.to_numpy(), bh.to_numpy())


def test_daily_rebalanced_equals_average_return():
    p = _prices()
    tgt = pd.DataFrame(0.5, index=p.index, columns=p.columns)
    res = run_backtest(p, tgt, CostModel.zero(), execution_lag=0)
    expected = p.pct_change().fillna(0).mean(axis=1).to_numpy()
    np.testing.assert_allclose(res.returns.to_numpy(), expected, atol=1e-15)


def test_short_position_accounting():
    p = _prices()
    tgt = pd.DataFrame({"A": [-1.0]}, index=p.index[:1])
    res = run_backtest(p, tgt, CostModel.zero(), execution_lag=0)
    # NAV = 2 - P_t / P_0 for a short funded with cash (cash = 2)
    np.testing.assert_allclose(res.equity.to_numpy(), (2 - p["A"] / 100).to_numpy())


def test_execution_lag_delays_exposure():
    p = _prices()
    tgt = pd.DataFrame({"A": [1.0]}, index=p.index[:1])
    lag1 = run_backtest(p, tgt, CostModel.zero(), execution_lag=1)
    assert lag1.returns.iloc[1] == 0.0
    assert lag1.returns.iloc[2] == pytest.approx(99 / 110 - 1)


def test_costs_and_borrow():
    p = _prices()
    c = CostModel(commission_bps=5, slippage_bps=5, borrow_bps_annual=252)  # 1bp/day borrow
    tgt = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=p.index[:1])
    res = run_backtest(p, tgt, c, execution_lag=0)
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.returns.iloc[0] == pytest.approx(-0.001)
    zero = run_backtest(p, tgt, CostModel.zero(), execution_lag=0)
    assert (res.costs.iloc[1:] > 0).all()
    np.testing.assert_allclose(res.gross_returns - res.returns, res.costs)
    assert res.equity.iloc[-1] < zero.equity.iloc[-1]


def test_targets_on_change():
    idx = pd.bdate_range("2021-01-01", periods=4)
    t = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0]}, index=idx)
    out = targets_on_change(t)
    assert out["A"].isna().tolist() == [False, True, False, True]


def test_input_validation():
    p = _prices()
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame({"C": [1.0]}, index=p.index[:1]))
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-02")]))
    bad = p.copy()
    bad.iloc[2, 0] = np.nan
    with pytest.raises(ValueError):
        run_backtest(bad, pd.DataFrame({"A": [1.0]}, index=p.index[:1]))
