"""High-level runners that wire signals, the backtest engine and reporting together."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .allocation import combine_sleeves, run_allocation
from .backtest import BacktestResult, run_backtest, targets_on_change
from .config import AllocationConfig, CostModel, PairsConfig
from .pairs import PairsResult, walk_forward_pairs


def run_pairs_strategy(
    prices: pd.DataFrame,
    cfg: PairsConfig,
    costs: CostModel | None = None,
    execution_lag: int = 1,
) -> tuple[BacktestResult, PairsResult]:
    """Walk-forward pairs targets -> backtest, trimmed to the first trading window."""
    pr = walk_forward_pairs(prices, cfg)
    # Hold share positions between signal changes instead of re-trading drift daily.
    targets = targets_on_change(pr.targets)
    bt = run_backtest(prices, targets, costs, execution_lag=execution_lag)
    return bt.trim(pr.first_trade_date), pr


@dataclass
class CompositeResult:
    tactical: BacktestResult
    strategic: BacktestResult
    combined: BacktestResult
    pairs: PairsResult

    def summary_table(self, benchmark: pd.Series | None = None) -> pd.DataFrame:
        start = max(self.tactical.returns.index[0], self.strategic.returns.index[0])
        rows = {
            "tactical_pairs": self.tactical.trim(start).summary(benchmark),
            "strategic_allocation": self.strategic.trim(start).summary(benchmark),
            "combined": self.combined.trim(start).summary(benchmark),
        }
        return pd.DataFrame(rows)


def run_composite(
    pairs_prices: pd.DataFrame,
    alloc_prices: pd.DataFrame,
    pairs_cfg: PairsConfig,
    alloc_cfg: AllocationConfig,
    tactical_weight: float = 0.3,
    costs: CostModel | None = None,
    sleeve_rebalance_every: int = 21,
) -> CompositeResult:
    """Run the tactical (pairs) and strategic (allocation) sleeves and combine them."""
    if not 0 <= tactical_weight <= 1:
        raise ValueError("tactical_weight must be in [0, 1]")
    tactical, pr = run_pairs_strategy(pairs_prices, pairs_cfg, costs)
    strategic = run_allocation(alloc_prices, alloc_cfg, costs)
    start = max(tactical.returns.index[0], strategic.returns.index[0])
    end = min(tactical.returns.index[-1], strategic.returns.index[-1])
    if start >= end:
        raise ValueError("sleeves do not overlap in time")
    sl = slice(start, end)
    combined = combine_sleeves(
        {"tactical": tactical.returns.loc[sl], "strategic": strategic.returns.loc[sl]},
        {"tactical": tactical_weight, "strategic": 1.0 - tactical_weight},
        rebalance_every=sleeve_rebalance_every,
    )
    return CompositeResult(tactical, strategic, combined, pr)
