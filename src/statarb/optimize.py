"""Long-only, fully invested portfolio construction rules.

All functions take annualised or periodic inputs consistently (the scale
does not change the optimal weights) and return a numpy array of weights
that sums to one with every weight in [0, max_weight].
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

_TOL = 1e-8


def _check_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.isfinite(cov).all():
        raise ValueError("cov must be finite")
    if np.any(np.diag(cov) <= 0):
        raise ValueError("cov must have a strictly positive diagonal")
    return 0.5 * (cov + cov.T)


def _check_cap(n: int, max_weight: float) -> None:
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * n < 1 - _TOL:
        raise ValueError(f"infeasible: {n} assets * max_weight {max_weight} < 1")


def _finalize(w: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, max_weight)
    s = w.sum()
    if s <= 0:
        raise RuntimeError("optimizer produced all-zero weights")
    w = w / s
    if w.max() > max_weight + 1e-6:
        raise RuntimeError("optimizer violated the weight cap after normalisation")
    return w


def equal_weight(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    cov = _check_cov(cov)
    iv = 1.0 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def _solve_slsqp(fun, jac, n: int, max_weight: float, starts: list[np.ndarray]) -> np.ndarray:
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}]
    best_w, best_f = None, np.inf
    for x0 in starts:
        res = minimize(
            fun,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        w = res.x
        feasible = abs(w.sum() - 1.0) < 1e-6 and w.min() > -1e-8 and w.max() < max_weight + 1e-8
        if feasible and res.fun < best_f:
            best_w, best_f = w, res.fun
    if best_w is None:
        raise RuntimeError("SLSQP failed to find a feasible solution from any start")
    return best_w


def _starts(n: int, max_weight: float, n_restarts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(rng.dirichlet(np.ones(n)))
    return starts


def min_variance(cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    cov = _check_cov(cov)
    n = cov.shape[0]
    _check_cap(n, max_weight)
    scale = np.trace(cov) / n  # conditioning only; optimum is scale-invariant
    c = cov / scale

    def f(w):
        return float(w @ c @ w)

    def g(w):
        return 2.0 * c @ w

    # convex problem: a single start suffices
    return _finalize(_solve_slsqp(f, g, n, max_weight, [np.full(n, 1.0 / n)]), max_weight)


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    max_weight: float = 1.0,
    n_restarts: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Maximise (w'mu - rf) / sqrt(w' cov w) with multi-start SLSQP.

    If no asset has an expected return above rf the tangency portfolio is
    not defined; the minimum-variance portfolio is returned instead.
    """
    cov = _check_cov(cov)
    mu = np.asarray(mu, dtype=float)
    n = cov.shape[0]
    if mu.shape != (n,) or not np.isfinite(mu).all():
        raise ValueError("mu must be a finite vector matching cov")
    _check_cap(n, max_weight)
    excess = mu - rf
    if excess.max() <= 0:
        warnings.warn("no asset beats rf; falling back to min-variance", stacklevel=2)
        return min_variance(cov, max_weight)

    def f(w):
        sd = np.sqrt(max(w @ cov @ w, 1e-18))
        return -float(w @ excess) / sd

    def g(w):
        var = max(w @ cov @ w, 1e-18)
        sd = np.sqrt(var)
        ret = w @ excess
        return -(excess * sd - ret * (cov @ w) / sd) / var

    starts = _starts(n, max_weight, n_restarts, seed)
    return _finalize(_solve_slsqp(f, g, n, max_weight, starts), max_weight)


def risk_parity(cov: np.ndarray, budgets: np.ndarray | None = None) -> np.ndarray:
    """Equal (or budgeted) risk contribution portfolio.

    Solves the strictly convex problem of Spinu (2013):
        min_y 0.5 y' cov y - sum_i b_i ln y_i,  y > 0,
    whose solution normalised to sum one has risk contributions
    w_i (cov w)_i / (w' cov w) = b_i.
    """
    cov = _check_cov(cov)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if b.shape != (n,) or np.any(b <= 0):
        raise ValueError("budgets must be a positive vector matching cov")
    b = b / b.sum()
    c = cov / (np.trace(cov) / n)

    def f(y):
        return float(0.5 * y @ c @ y - b @ np.log(y))

    def g(y):
        return c @ y - b / y

    x0 = 1.0 / np.sqrt(np.diag(c))
    x0 = x0 / np.sqrt(x0 @ c @ x0)
    res = minimize(
        f,
        x0,
        jac=g,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"maxiter": 5000, "gtol": 1e-12, "ftol": 1e-15},
    )
    y = res.x
    if not np.all(np.isfinite(y)) or y.sum() <= 0:
        raise RuntimeError("risk parity optimisation diverged")
    # Normalise before _finalize: y is unscaled and can exceed 1 elementwise.
    return _finalize(y / y.sum(), 1.0)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fractional risk contributions w_i (cov w)_i / (w' cov w); sums to 1."""
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    total = w @ cov @ w
    return w * (cov @ w) / total
