"""Configuration objects. All parameters are validated on construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    """Linear transaction-cost model.

    ``commission_bps`` and ``slippage_bps`` are charged on traded notional
    (turnover) in basis points of NAV. ``borrow_bps_annual`` accrues daily on
    gross short exposure.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 50.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "borrow_bps_annual"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def trade_cost_rate(self) -> float:
        """Cost per unit of turnover, as a fraction of NAV."""
        return (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def daily_borrow_rate(self) -> float:
        return self.borrow_bps_annual / 1e4 / TRADING_DAYS

    @classmethod
    def zero(cls) -> CostModel:
        return cls(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PairsConfig:
    """Walk-forward pairs-trading parameters."""

    formation_days: int = 252
    trading_days: int = 63
    pvalue_threshold: float = 0.05
    fdr_control: bool = True  # Benjamini-Hochberg across all tested pairs
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    max_pairs: int = 5
    gross_leverage: float = 1.0  # each selected pair gets gross_leverage / max_pairs
    min_correlation: float | None = None  # optional pre-filter on daily log-return corr
    hedge_method: Literal["ols", "kalman"] = "ols"
    zscore_lookback: int | None = None  # None -> formation-window mean/std (OLS only)
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_holding_multiple: float | None = 3.0  # time stop = multiple * half-life (days)
    # State-noise scale. On log prices (x ~ 3-6) the per-step state noise x^2 * delta must
    # be small next to the spread variance (~1e-4..1e-3) or the filter absorbs the spread
    # into beta/alpha and z-scores collapse. 1e-4 (a common default for price levels)
    # is far too large here.
    kalman_delta: float = 1e-7
    kalman_obs_var: float | None = None  # None -> residual variance of formation OLS

    def __post_init__(self) -> None:
        if self.formation_days < 30:
            raise ValueError("formation_days must be >= 30")
        if self.trading_days < 1:
            raise ValueError("trading_days must be >= 1")
        if not 0 < self.pvalue_threshold < 1:
            raise ValueError("pvalue_threshold must be in (0, 1)")
        if not 0 < self.min_half_life < self.max_half_life:
            raise ValueError("require 0 < min_half_life < max_half_life")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")
        if not 0 < self.gross_leverage <= 10:
            raise ValueError("gross_leverage must be in (0, 10]")
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        if self.hedge_method not in ("ols", "kalman"):
            raise ValueError("hedge_method must be 'ols' or 'kalman'")
        if self.zscore_lookback is not None:
            if self.zscore_lookback < 5:
                raise ValueError("zscore_lookback must be >= 5 or None")
            if self.zscore_lookback > self.formation_days:
                raise ValueError("zscore_lookback cannot exceed formation_days")
        if self.max_holding_multiple is not None and self.max_holding_multiple <= 0:
            raise ValueError("max_holding_multiple must be positive or None")
        if not 0 < self.kalman_delta < 1:
            raise ValueError("kalman_delta must be in (0, 1)")
        if self.kalman_obs_var is not None and self.kalman_obs_var <= 0:
            raise ValueError("kalman_obs_var must be positive or None")
        if self.min_correlation is not None and not -1 <= self.min_correlation <= 1:
            raise ValueError("min_correlation must be in [-1, 1]")


AllocationMethod = Literal["equal", "inverse_vol", "min_variance", "max_sharpe", "risk_parity"]
CovarianceMethod = Literal["sample", "ledoit_wolf", "ewma"]

ALLOCATION_METHODS = ("equal", "inverse_vol", "min_variance", "max_sharpe", "risk_parity")
COVARIANCE_METHODS = ("sample", "ledoit_wolf", "ewma")


@dataclass(frozen=True)
class AllocationConfig:
    """Rolling (walk-forward) strategic allocation parameters."""

    method: AllocationMethod = "risk_parity"
    lookback_days: int = 252
    rebalance_every: int = 21
    covariance: CovarianceMethod = "ledoit_wolf"
    ewma_halflife: int = 63
    max_weight: float = 1.0  # ignored by risk_parity (ERC solution is unique)
    risk_free_rate: float = 0.0  # annualised, used by max_sharpe
    mean_shrinkage: float = 0.5  # shrink expected returns toward the cross-sectional mean
    n_restarts: int = 10
    trend_filter: bool = False  # send assets with negative trailing return to cash
    trend_lookback: int = 252
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in ALLOCATION_METHODS:
            raise ValueError(f"method must be one of {ALLOCATION_METHODS}")
        if self.covariance not in COVARIANCE_METHODS:
            raise ValueError(f"covariance must be one of {COVARIANCE_METHODS}")
        if self.lookback_days < 20:
            raise ValueError("lookback_days must be >= 20")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 <= self.mean_shrinkage <= 1:
            raise ValueError("mean_shrinkage must be in [0, 1]")
        if self.n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        if self.ewma_halflife < 1:
            raise ValueError("ewma_halflife must be >= 1")
        if self.trend_lookback < 2:
            raise ValueError("trend_lookback must be >= 2")
