"""Statistical building blocks: OLS hedge ratios, cointegration, half-life, FDR."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of y on [1, x]. Returns (alpha, beta)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or y.shape != x.shape:
        raise ValueError("y and x must be 1-D arrays of equal length")
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


@dataclass(frozen=True)
class CointegrationResult:
    tstat: float
    pvalue: float
    alpha: float
    beta: float
    residual_std: float


def engle_granger(y: np.ndarray, x: np.ndarray) -> CointegrationResult:
    """Engle-Granger two-step test with y regressed on x (constant included).

    p-values are MacKinnon approximations computed by statsmodels. The test is
    not symmetric in (y, x), so callers must fix an ordering convention.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tstat, pvalue, _ = coint(y, x, trend="c", autolag="aic")
    alpha, beta = ols_hedge_ratio(y, x)
    resid = y - alpha - beta * x
    return CointegrationResult(float(tstat), float(pvalue), alpha, beta, float(resid.std(ddof=1)))


def half_life(spread: np.ndarray) -> float:
    """Mean-reversion half-life in periods, from an AR(1) fit.

    Fits s_t = c + phi * s_{t-1} + e_t, half-life = -ln(2) / ln(phi).
    Returns inf when phi >= 1 (no mean reversion) and nan when phi <= 0
    (oscillating, not an OU-type process).
    """
    s = np.asarray(spread, dtype=float)
    if s.ndim != 1 or len(s) < 10:
        raise ValueError("spread must be 1-D with at least 10 observations")
    lag, cur = s[:-1], s[1:]
    X = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(X, cur, rcond=None)
    phi = float(coef[1])
    if phi >= 1.0:
        return float("inf")
    if phi <= 0.0:
        return float("nan")
    return float(-np.log(2.0) / np.log(phi))


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Boolean rejection mask controlling the false discovery rate at level q."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = int(np.max(np.nonzero(passed)[0]))
        reject[order[: k + 1]] = True
    return reject


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score: the value at t uses observations t-window+1..t only."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)
