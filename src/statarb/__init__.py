"""statarb: walk-forward pairs trading and strategic asset allocation."""

from .allocation import allocation_targets, combine_sleeves, compute_weights, run_allocation
from .backtest import BacktestResult, run_backtest, targets_on_change
from .config import AllocationConfig, CostModel, PairsConfig
from .data import clean_prices, load_csv, load_yahoo, simulate_prices
from .metrics import performance_summary
from .pairs import find_pairs, generate_positions, walk_forward_pairs
from .strategy import CompositeResult, run_composite, run_pairs_strategy

__version__ = "0.1.0"

__all__ = [
    "AllocationConfig",
    "BacktestResult",
    "CompositeResult",
    "CostModel",
    "PairsConfig",
    "allocation_targets",
    "clean_prices",
    "combine_sleeves",
    "compute_weights",
    "find_pairs",
    "generate_positions",
    "load_csv",
    "load_yahoo",
    "performance_summary",
    "run_allocation",
    "run_backtest",
    "run_composite",
    "run_pairs_strategy",
    "simulate_prices",
    "targets_on_change",
    "walk_forward_pairs",
]
