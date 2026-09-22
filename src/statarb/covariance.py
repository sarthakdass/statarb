"""Covariance estimators operating on a (T x N) matrix of periodic returns."""

from __future__ import annotations

import numpy as np


def _as_2d(returns: np.ndarray) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("returns must be 2-D with at least 2 rows")
    if not np.isfinite(X).all():
        raise ValueError("returns must be finite")
    return X


def sample_cov(returns: np.ndarray) -> np.ndarray:
    X = _as_2d(returns)
    return np.atleast_2d(np.cov(X, rowvar=False, ddof=1)).reshape(X.shape[1], X.shape[1])


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage toward a scaled identity.

    Returns (covariance, shrinkage intensity in [0, 1]). Matches the
    scikit-learn estimator (MLE normalisation by T, data demeaned).
    """
    X = _as_2d(returns)
    X = X - X.mean(axis=0)
    n, p = X.shape
    emp = X.T @ X / n
    mu = np.trace(emp) / p
    X2 = X**2
    beta_ = float(np.sum(X2.T @ X2)) / n
    delta_ = float(np.sum(emp**2))
    beta = (beta_ / n - delta_ / n) / p  # = (1/(n p)) * (sum ||x x' - S||^2 / n)
    delta = (delta_ - 2.0 * mu * np.trace(emp) + p * mu**2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if delta == 0 else beta / delta
    cov = (1.0 - shrinkage) * emp + shrinkage * mu * np.eye(p)
    return cov, float(shrinkage)


def ewma_cov(returns: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance; the most recent row gets the largest weight."""
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    X = _as_2d(returns)
    n = X.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    mean = w @ X
    Xc = X - mean
    return (Xc * w[:, None]).T @ Xc / (1.0 - np.sum(w**2))  # bias-corrected weights


def estimate_cov(returns: np.ndarray, method: str, ewma_halflife: float = 63) -> np.ndarray:
    if method == "sample":
        cov = sample_cov(returns)
    elif method == "ledoit_wolf":
        cov, _ = ledoit_wolf(returns)
    elif method == "ewma":
        cov = ewma_cov(returns, ewma_halflife)
    else:
        raise ValueError(f"unknown covariance method: {method}")
    return 0.5 * (cov + cov.T)
