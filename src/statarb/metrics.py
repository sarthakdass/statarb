"""Performance and risk metrics computed on periodic simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("returns are empty")
    return r


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric (compounded) annualised return."""
    r = _clean(returns)
    growth = float(np.prod(1.0 + r.to_numpy()))
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(r)) - 1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of per-period excess returns (rf is annual)."""
    r = _clean(returns) - rf / periods
    sd = r.std(ddof=1)
    if len(r) < 2 or not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = _clean(returns) - rf / periods
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    if downside == 0:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + _clean(returns)).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)  # starting capital counts as a peak
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough decline of compounded equity (<= 0)."""
    return float(drawdown_series(returns).min())


def beta_alpha(
    returns: pd.Series, benchmark: pd.Series, periods: int = TRADING_DAYS
) -> tuple[float, float]:
    """OLS beta and annualised (arithmetic) alpha versus a benchmark."""
    df = pd.concat([_clean(returns), pd.Series(benchmark, dtype=float)], axis=1, join="inner")
    df = df.dropna()
    if len(df) < 3:
        return float("nan"), float("nan")
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    var_x = x.var(ddof=1)
    if var_x == 0:
        return float("nan"), float("nan")
    beta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    alpha = float((y.mean() - beta * x.mean()) * periods)
    return beta, alpha


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Bailey & Lopez de Prado (2012) PSR: P(true Sharpe > benchmark_sharpe).

    ``benchmark_sharpe`` is annualised; the test uses per-period moments and
    adjusts for skewness and kurtosis of the return distribution.
    """
    r = _clean(returns)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0 or not np.isfinite(sd):
        return float("nan")
    sr = r.mean() / sd
    sr_star = benchmark_sharpe / np.sqrt(periods)
    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)))


def performance_summary(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    r = _clean(returns)
    ann_ret = annualized_return(r, periods)
    mdd = max_drawdown(r)
    active = r[r != 0]
    out = {
        "start": r.index[0],
        "end": r.index[-1],
        "n_periods": len(r),
        "total_return": float(np.prod(1.0 + r.to_numpy()) - 1.0),
        "ann_return": ann_ret,
        "ann_vol": annualized_volatility(r, periods),
        "sharpe": sharpe_ratio(r, rf, periods),
        "sortino": sortino_ratio(r, rf, periods),
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else float("nan"),
        "psr_vs_0": probabilistic_sharpe_ratio(r, 0.0, periods),
        "skew": float(stats.skew(r, bias=False)) if len(r) > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(r, bias=False)) if len(r) > 3 else float("nan"),
        "hit_rate_active_days": float((active > 0).mean()) if len(active) else float("nan"),
    }
    if benchmark is not None:
        b, a = beta_alpha(r, benchmark, periods)
        out["beta"] = b
        out["alpha_ann"] = a
        joined = pd.concat([r, pd.Series(benchmark, dtype=float)], axis=1, join="inner").dropna()
        out["corr_benchmark"] = float(joined.corr().iloc[0, 1]) if len(joined) > 2 else float("nan")
    return pd.Series(out)
