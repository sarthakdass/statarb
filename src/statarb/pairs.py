"""Pairs trading: pair selection, spread signals and walk-forward target weights.

Timing convention (used everywhere in this package): a target produced for
date t uses only prices up to and including the close of t. The backtest
engine then executes it with a configurable lag (default: the next close).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .config import PairsConfig
from .kalman import kalman_hedge
from .stats import benjamini_hochberg, engle_granger, half_life, rolling_zscore


@dataclass(frozen=True)
class PairCandidate:
    """A pair that passed the formation-window screen. y is the dependent leg."""

    y: str
    x: str
    pvalue: float
    alpha: float
    beta: float
    half_life: float
    spread_mean: float
    spread_std: float


def find_pairs(prices: pd.DataFrame, cfg: PairsConfig) -> list[PairCandidate]:
    """Screen all pairs in a formation window for cointegration.

    Convention: for tickers (a, b) in sorted order, a is regressed on b.
    Selection rules, in order:
      1. Engle-Granger p-value, with Benjamini-Hochberg FDR control across all
         tested pairs if ``cfg.fdr_control`` (otherwise a raw threshold);
      2. positive hedge ratio (a genuine long/short spread);
      3. half-life within [min_half_life, max_half_life];
    then the ``max_pairs`` lowest p-values are kept.
    """
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("prices must be positive with no missing values")
    logp = np.log(prices)
    tickers = sorted(logp.columns)

    corr = None
    if cfg.min_correlation is not None:
        corr = logp.diff().dropna().corr()

    tested: list[tuple[str, str]] = []
    results = []
    for a, b in combinations(tickers, 2):
        if corr is not None and corr.loc[a, b] < cfg.min_correlation:
            continue
        res = engle_granger(logp[a].to_numpy(), logp[b].to_numpy())
        tested.append((a, b))
        results.append(res)
    if not tested:
        return []

    pvals = np.array([r.pvalue for r in results])
    if cfg.fdr_control:
        significant = benjamini_hochberg(pvals, cfg.pvalue_threshold)
    else:
        significant = pvals < cfg.pvalue_threshold

    candidates: list[PairCandidate] = []
    for (a, b), res, sig in zip(tested, results, significant, strict=True):
        if not sig or res.beta <= 0:
            continue
        spread = logp[a].to_numpy() - res.alpha - res.beta * logp[b].to_numpy()
        hl = half_life(spread)
        if not (np.isfinite(hl) and cfg.min_half_life <= hl <= cfg.max_half_life):
            continue
        candidates.append(
            PairCandidate(
                y=a,
                x=b,
                pvalue=res.pvalue,
                alpha=res.alpha,
                beta=res.beta,
                half_life=hl,
                spread_mean=float(spread.mean()),
                spread_std=float(spread.std(ddof=1)),
            )
        )
    candidates.sort(key=lambda c: c.pvalue)
    return candidates[: cfg.max_pairs]


def generate_positions(
    z: np.ndarray,
    entry: float,
    exit_: float,
    stop: float,
    max_hold: int | None = None,
) -> np.ndarray:
    """Convert a z-score path into spread positions in {-1, 0, +1}.

    +1 = long spread (long y, short x), entered when z <= -entry.
    -1 = short spread, entered when z >= +entry.
    Exit when z reverts through the exit band, on a stop-loss (|z| >= stop)
    or after ``max_hold`` bars. After a stop or time exit the rule is
    disarmed until |z| falls back below ``entry``, so it cannot immediately
    re-enter a spread that just failed. Non-finite z flattens the position.
    Decisions at bar t use z[t] only (no future information).
    """
    if not 0 <= exit_ < entry < stop:
        raise ValueError("require 0 <= exit < entry < stop")
    z = np.asarray(z, dtype=float)
    pos = np.zeros(len(z))
    cur = 0
    held = 0
    armed = True
    for t, zt in enumerate(z):
        if not np.isfinite(zt):
            cur, held = 0, 0
            pos[t] = 0
            continue
        if not armed and abs(zt) < entry:
            armed = True
        if cur == 0:
            if armed and entry <= zt < stop:
                cur, held = -1, 0
            elif armed and -stop < zt <= -entry:
                cur, held = 1, 0
        else:
            held += 1
            reverted = (cur == 1 and zt >= -exit_) or (cur == -1 and zt <= exit_)
            stopped = abs(zt) >= stop
            timed_out = max_hold is not None and held >= max_hold
            if reverted or stopped or timed_out:
                if (stopped or timed_out) and not reverted:
                    armed = False
                cur, held = 0, 0
        pos[t] = cur
    return pos


def spread_to_weights(
    position: np.ndarray, beta: np.ndarray, capital: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dollar weights for (y, x) holding ``position`` units of y - beta * x.

    The log-spread changes by r_y - beta * r_x, so dollar exposures are
    proportional to (1, -beta); they are scaled to gross exposure ``capital``.
    """
    beta = np.asarray(beta, dtype=float)
    gross = 1.0 + np.abs(beta)
    w_y = position * capital / gross
    w_x = -position * beta * capital / gross
    return w_y, w_x


@dataclass
class PairsResult:
    targets: pd.DataFrame  # daily target weights, all assets
    selections: pd.DataFrame  # one row per (window, selected pair)
    spread_positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    first_trade_date: pd.Timestamp | None = None


def _pair_signal(
    logp: pd.DataFrame, pair: PairCandidate, cfg: PairsConfig, formation_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """z-score and hedge ratio over the trading part of a formation+trading slice."""
    y, x = logp[pair.y], logp[pair.x]
    if cfg.hedge_method == "kalman":
        obs_var = cfg.kalman_obs_var if cfg.kalman_obs_var is not None else pair.spread_std**2
        kf = kalman_hedge(
            y, x, delta=cfg.kalman_delta, obs_var=obs_var, init_state=(pair.beta, pair.alpha)
        )
        z = kf["zscore"].to_numpy()[formation_len:]
        beta = kf["beta"].to_numpy()[formation_len:]
        return z, beta

    spread = y - pair.alpha - pair.beta * x
    if cfg.zscore_lookback is None:
        z_all = (spread - pair.spread_mean) / pair.spread_std
    else:
        z_all = rolling_zscore(spread, cfg.zscore_lookback)
    z = z_all.to_numpy()[formation_len:]
    beta = np.full(len(z), pair.beta)
    return z, beta


def walk_forward_pairs(prices: pd.DataFrame, cfg: PairsConfig) -> PairsResult:
    """Rolling formation/trading backtest targets.

    For each window starting at index s: pairs are selected on prices
    [s - formation_days, s) and traded on [s, s + trading_days). Each selected
    pair receives gross capital gross_leverage / max_pairs, so total gross
    exposure never exceeds gross_leverage, however many pairs pass the screen. Positions are
    flattened when a pair drops out of the next window's selection.
    """
    prices = prices.sort_index()
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("prices must be positive with no missing values; use clean_prices")
    n = len(prices)
    F, T = cfg.formation_days, cfg.trading_days
    if n <= F:
        raise ValueError(f"need more than formation_days={F} observations, got {n}")

    logp = np.log(prices)
    targets = np.zeros(prices.shape)
    col_idx = {c: i for i, c in enumerate(prices.columns)}
    capital = cfg.gross_leverage / cfg.max_pairs
    records = []
    spread_cols: dict[str, np.ndarray] = {}

    for start in range(F, n, T):
        stop = min(start + T, n)
        formation = prices.iloc[start - F : start]
        pairs = find_pairs(formation, cfg)
        for p in pairs:
            records.append(
                {
                    "window_start": prices.index[start],
                    "window_end": prices.index[stop - 1],
                    "y": p.y,
                    "x": p.x,
                    "pvalue": p.pvalue,
                    "beta": p.beta,
                    "half_life": p.half_life,
                }
            )
            z, beta = _pair_signal(logp.iloc[start - F : stop], p, cfg, F)
            max_hold = (
                None
                if cfg.max_holding_multiple is None
                else max(1, math.ceil(cfg.max_holding_multiple * p.half_life))
            )
            pos = generate_positions(z, cfg.entry_z, cfg.exit_z, cfg.stop_z, max_hold)
            w_y, w_x = spread_to_weights(pos, beta, capital)
            targets[start:stop, col_idx[p.y]] += w_y
            targets[start:stop, col_idx[p.x]] += w_x

            key = f"{p.y}/{p.x}"
            if key not in spread_cols:
                spread_cols[key] = np.zeros(n)
            spread_cols[key][start:stop] = pos

    selections = pd.DataFrame(
        records, columns=["window_start", "window_end", "y", "x", "pvalue", "beta", "half_life"]
    )
    return PairsResult(
        targets=pd.DataFrame(targets, index=prices.index, columns=prices.columns),
        selections=selections,
        spread_positions=pd.DataFrame(spread_cols, index=prices.index),
        first_trade_date=prices.index[F],
    )
