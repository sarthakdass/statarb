"""Walk-forward strategic allocation and multi-sleeve portfolio combination."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import TRADING_DAYS, AllocationConfig, CostModel
from .covariance import estimate_cov
from .optimize import (
    equal_weight,
    inverse_volatility,
    max_sharpe,
    min_variance,
    risk_parity,
)


def shrink_means(mu: np.ndarray, intensity: float) -> np.ndarray:
    """Shrink expected returns toward their cross-sectional average (James-Stein style)."""
    if not 0 <= intensity <= 1:
        raise ValueError("intensity must be in [0, 1]")
    return (1.0 - intensity) * mu + intensity * mu.mean()


def compute_weights(window_returns: pd.DataFrame, cfg: AllocationConfig) -> np.ndarray:
    """Weights from a window of simple daily returns (rows = days, cols = assets)."""
    R = window_returns.to_numpy(dtype=float)
    n = R.shape[1]
    if cfg.method == "equal":
        return equal_weight(n)
    cov = estimate_cov(R, cfg.covariance, cfg.ewma_halflife) * TRADING_DAYS
    if cfg.method == "inverse_vol":
        return inverse_volatility(cov)
    if cfg.method == "min_variance":
        return min_variance(cov, cfg.max_weight)
    if cfg.method == "risk_parity":
        return risk_parity(cov)
    if cfg.method == "max_sharpe":
        mu = np.log1p(R).mean(axis=0) * TRADING_DAYS  # log mean is robust to vol drag
        mu = shrink_means(mu, cfg.mean_shrinkage)
        return max_sharpe(mu, cov, cfg.risk_free_rate, cfg.max_weight, cfg.n_restarts, cfg.seed)
    raise ValueError(f"unknown method {cfg.method}")


def allocation_targets(prices: pd.DataFrame, cfg: AllocationConfig) -> pd.DataFrame:
    """Target weights on rebalance dates only (other dates are absent -> drift).

    The target at date t uses returns ending at the close of t
    (the last ``lookback_days`` of them). Run the result through
    ``run_backtest`` with ``execution_lag >= 1``.
    """
    prices = prices.sort_index()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("prices must be positive with no missing values")
    rets = prices.pct_change()
    L = cfg.lookback_days
    first = max(L, cfg.trend_lookback if cfg.trend_filter else 0)
    if len(prices) <= first:
        raise ValueError("not enough history for the lookback")
    rows, dates = [], []
    for i in range(first, len(prices), cfg.rebalance_every):
        window = rets.iloc[i - L + 1 : i + 1]
        w = compute_weights(window, cfg)
        if cfg.trend_filter:
            trailing = prices.iloc[i] / prices.iloc[i - cfg.trend_lookback] - 1.0
            w = np.where(trailing.to_numpy() > 0, w, 0.0)  # remainder held in cash
        rows.append(w)
        dates.append(prices.index[i])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=prices.columns)


def run_allocation(
    prices: pd.DataFrame,
    cfg: AllocationConfig,
    costs: CostModel | None = None,
    execution_lag: int = 1,
) -> BacktestResult:
    targets = allocation_targets(prices, cfg)
    result = run_backtest(prices, targets, costs, execution_lag=execution_lag)
    return result.trim(targets.index[0])


def combine_sleeves(
    sleeve_returns: Mapping[str, pd.Series],
    sleeve_weights: Mapping[str, float],
    rebalance_every: int = 21,
) -> BacktestResult:
    """Combine independently simulated strategies ("sleeves") into one book.

    Each sleeve's net-of-cost equity curve is treated as a tradable asset and
    the book is rebalanced back to fixed sleeve weights every
    ``rebalance_every`` days, with drift in between. Sleeves are aligned on
    the union of dates; a sleeve earns 0 (cash) before it starts. Moving
    capital between sleeves is assumed costless because trading costs are
    already inside each sleeve.
    """
    if set(sleeve_returns) != set(sleeve_weights):
        raise ValueError("sleeve_returns and sleeve_weights must have the same keys")
    w = pd.Series(sleeve_weights, dtype=float)
    if (w < 0).any() or w.sum() > 1 + 1e-9:
        raise ValueError("sleeve weights must be non-negative and sum to <= 1")
    rets = pd.DataFrame(dict(sleeve_returns)).sort_index().fillna(0.0)
    navs = (1.0 + rets).cumprod()
    navs = navs / navs.iloc[0]
    if (navs <= 0).any().any():
        raise ValueError("a sleeve NAV is non-positive")
    dates = navs.index[::rebalance_every]
    targets = pd.DataFrame(
        [w.reindex(navs.columns).to_numpy()] * len(dates), index=dates, columns=navs.columns
    )
    # Weights are fixed constants (no information), so no execution lag is needed.
    return run_backtest(navs, targets, CostModel.zero(), execution_lag=0)
