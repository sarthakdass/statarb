"""Event-free daily backtest engine with weight drift, execution lag and costs.

Accounting (all weights are fractions of NAV, cash = 1 - sum(weights)):

* Over day t (close t-1 -> close t), with asset simple returns r_t,
  gross return  g_t = w . r_t + cash * rf_daily
  borrow cost   b_t = sum(max(-w, 0)) * borrow_daily
  weights drift to w_i (1 + r_i) / (1 + g_t - b_t).
  Short positions drift correctly under this rule because the short
  proceeds sit in cash.
* At the close of t, if a (lagged) target exists, trade to it:
  turnover u_t = sum |target - w|, cost c_t = u_t * trade_cost_rate.
* Net return for day t: (1 + g_t - b_t) * (1 - c_t) - 1.

Target rows that are entirely NaN mean "no rebalance, let weights drift".
In a row that is not entirely NaN, missing assets are treated as 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, CostModel
from .metrics import performance_summary


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily returns
    gross_returns: pd.Series  # before trading and borrow costs
    costs: pd.Series  # trading + borrow cost as a fraction of NAV
    turnover: pd.Series  # one-way turnover, fraction of NAV
    weights: pd.DataFrame  # post-trade weights at each close

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def trim(self, start: pd.Timestamp | str | None) -> BacktestResult:
        """Drop the warm-up period before ``start`` (inclusive start)."""
        if start is None:
            return self
        sl = slice(pd.Timestamp(start), None)
        return BacktestResult(
            self.returns.loc[sl],
            self.gross_returns.loc[sl],
            self.costs.loc[sl],
            self.turnover.loc[sl],
            self.weights.loc[sl],
        )

    def summary(self, benchmark: pd.Series | None = None, rf: float = 0.0) -> pd.Series:
        s = performance_summary(self.returns, benchmark=benchmark, rf=rf)
        s["ann_turnover"] = float(self.turnover.mean() * TRADING_DAYS)
        s["ann_cost_drag"] = float(self.costs.mean() * TRADING_DAYS)
        s["avg_gross_exposure"] = float(self.weights.abs().sum(axis=1).mean())
        return s


def targets_on_change(targets: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """Keep only rows where the target differs from the previous row; others become NaN.

    Use this for signal-driven strategies so that unchanged targets are held
    as share positions (weights drift) instead of being re-traded daily.
    """
    diff = targets.diff().abs().max(axis=1)
    changed = diff > tol
    changed.iloc[0] = True
    out = targets.copy()
    out.loc[~changed, :] = np.nan
    return out


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    costs: CostModel | None = None,
    execution_lag: int = 1,
    rf_annual: float = 0.0,
) -> BacktestResult:
    """Simulate trading ``targets`` on ``prices``.

    ``execution_lag`` shifts targets forward by that many trading days, so a
    target computed from the close of t is executed at the close of t + lag.
    Use lag >= 1 for any signal computed from same-day closes.
    """
    costs = costs or CostModel()
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    prices = prices.sort_index()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("prices must be positive with no missing values")
    unknown = set(targets.columns) - set(prices.columns)
    if unknown:
        raise ValueError(f"targets reference assets without prices: {sorted(unknown)}")
    extra_dates = targets.index.difference(prices.index)
    if len(extra_dates):
        raise ValueError(f"{len(extra_dates)} target dates are not trading dates in prices")

    tgt = targets.reindex(index=prices.index, columns=prices.columns)
    tgt = tgt.shift(execution_lag)
    has_target = tgt.notna().any(axis=1).to_numpy()
    tgt_arr = tgt.fillna(0.0).to_numpy()
    if not np.isfinite(tgt_arr).all():
        raise ValueError("targets contain non-finite values")

    rets = prices.pct_change().fillna(0.0).to_numpy()
    n, k = rets.shape
    rf_d = rf_annual / TRADING_DAYS
    rate = costs.trade_cost_rate
    borrow_d = costs.daily_borrow_rate

    w = np.zeros(k)
    net = np.zeros(n)
    gross = np.zeros(n)
    cost = np.zeros(n)
    turn = np.zeros(n)
    w_hist = np.zeros((n, k))

    for t in range(n):
        g = 0.0
        b = 0.0
        if t > 0:
            r = rets[t]
            g = float(w @ r + (1.0 - w.sum()) * rf_d)
            b = float(np.clip(-w, 0.0, None).sum() * borrow_d)
            growth = 1.0 + g - b
            if growth <= 0:
                raise RuntimeError(f"NAV wiped out on {prices.index[t].date()}")
            w = w * (1.0 + r) / growth
        c = 0.0
        if has_target[t]:
            u = float(np.abs(tgt_arr[t] - w).sum())
            c = u * rate
            if c >= 1:
                raise RuntimeError("trading cost exceeds NAV; check targets and costs")
            turn[t] = u
            w = tgt_arr[t].copy()
        gross[t] = g
        net[t] = (1.0 + g - b) * (1.0 - c) - 1.0
        cost[t] = g - net[t]  # exactly b + c * (1 + g - b)
        w_hist[t] = w

    idx = prices.index
    return BacktestResult(
        returns=pd.Series(net, idx, name="net"),
        gross_returns=pd.Series(gross, idx, name="gross"),
        costs=pd.Series(cost, idx, name="costs"),
        turnover=pd.Series(turn, idx, name="turnover"),
        weights=pd.DataFrame(w_hist, idx, prices.columns),
    )
