"""Price loading, cleaning and synthetic market simulation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def clean_prices(
    prices: pd.DataFrame,
    max_ffill: int = 5,
    max_missing_frac: float = 0.2,
) -> pd.DataFrame:
    """Return a gap-free, date-aligned panel of strictly positive prices.

    Steps: parse and sort the index, drop duplicate dates, coerce to numeric,
    treat non-positive values as missing, drop columns missing more than
    ``max_missing_frac`` of observations, forward-fill short gaps (at most
    ``max_ffill`` consecutive days, never backward), then keep only dates where
    every remaining asset has a price (inner alignment).

    Forward-filling carries stale prices across holidays on different exchange
    calendars, which creates artificial zero returns; keep ``max_ffill`` small.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.where(df > 0)

    missing = df.isna().mean()
    dropped = missing[missing > max_missing_frac].index.tolist()
    if dropped:
        warnings.warn(f"Dropping sparse columns: {dropped}", stacklevel=2)
        df = df.drop(columns=dropped)
    if df.shape[1] == 0:
        raise ValueError("No columns left after dropping sparse series")

    df = df.ffill(limit=max_ffill).dropna(how="any")
    if len(df) < 2:
        raise ValueError("Fewer than 2 aligned observations after cleaning")
    return df.astype(float)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a wide CSV: first column is the date, other columns are prices per ticker."""
    return clean_prices(pd.read_csv(path, index_col=0))


def load_yahoo(tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download split- and dividend-adjusted closes from Yahoo Finance via ``yfinance``.

    Uses a maintained client rather than scraping cookie/crumb tokens. Check
    Yahoo's terms of use before relying on this beyond personal research.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is not installed: pip install 'statarb[data]'") from exc

    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("tickers must be non-empty")
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        warnings.warn(f"No data returned for: {missing}", stacklevel=2)
    return clean_prices(close[[t for t in tickers if t in close.columns]])


@dataclass(frozen=True)
class PairTruth:
    """Ground truth for a simulated cointegrated pair."""

    y: str
    x: str
    hedge_ratio: float
    half_life: float


def simulate_prices(
    n_days: int = 2000,
    n_pairs: int = 3,
    n_independent: int = 4,
    seed: int = 0,
    start: str = "2012-01-02",
    half_life_range: tuple[float, float] = (3.0, 10.0),
) -> tuple[pd.DataFrame, list[PairTruth]]:
    """Simulate a market with known cointegrated pairs plus independent assets.

    For each pair, log(B) is a random walk loaded on a common market factor and
    log(A) = alpha + beta * log(B) + s, where s is a stationary AR(1) spread
    (discretised Ornstein-Uhlenbeck) with a known half-life. Independent
    assets are drifting random walks with market beta, usable as an
    allocation universe.

    Note on power: an Engle-Granger test on a 252-day window has little power
    for half-lives much above ~10 days (the expected ADF t-statistic is then
    near -2.3 versus a ~-3.4 critical value), so the default range is short.
    """
    if n_days < 50:
        raise ValueError("n_days must be >= 50")
    if not 0 < half_life_range[0] <= half_life_range[1]:
        raise ValueError("invalid half_life_range")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    market = rng.normal(0.0003, 0.009, n_days)
    cols: dict[str, np.ndarray] = {}
    truth: list[PairTruth] = []

    for i in range(n_pairs):
        mkt_beta = rng.uniform(0.6, 1.2)
        r_b = mkt_beta * market + rng.normal(0.0001, 0.011, n_days)
        r_b[0] = 0.0
        log_b = np.log(rng.uniform(20, 200)) + np.cumsum(r_b)
        hedge = rng.uniform(0.6, 1.4)
        hl = rng.uniform(*half_life_range)
        phi = 0.5 ** (1.0 / hl)
        eps = rng.normal(0.0, 0.008, n_days)
        s = np.empty(n_days)
        s[0] = eps[0] / np.sqrt(1.0 - phi**2)
        for t in range(1, n_days):
            s[t] = phi * s[t - 1] + eps[t]
        alpha = rng.uniform(-0.5, 0.5)
        log_a = alpha + hedge * log_b + s
        cols[f"A{i}"] = np.exp(log_a)
        cols[f"B{i}"] = np.exp(log_b)
        truth.append(PairTruth(f"A{i}", f"B{i}", hedge, hl))

    for j in range(n_independent):
        beta = rng.uniform(0.3, 1.2)
        r = beta * market + rng.normal(0.0002, 0.008 + 0.004 * j, n_days)
        r[0] = 0.0
        cols[f"IDX{j}"] = 100.0 * np.exp(np.cumsum(r))

    return pd.DataFrame(cols, index=dates), truth
