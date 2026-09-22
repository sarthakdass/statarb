"""Kalman filter for a time-varying hedge ratio.

Observation:  y_t = beta_t * x_t + alpha_t + e_t,        e_t ~ N(0, R)
State:        [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t,  w_t ~ N(0, Q)
with Q = delta / (1 - delta) * I (random-walk coefficients).

At each step the one-step-ahead forecast error (innovation) e_t is computed
with the *prior* state, which only uses information up to t-1, and its
variance S_t gives a natural z-score e_t / sqrt(S_t). The posterior state
(which also uses y_t, known at the close of t) is returned as the hedge ratio.
Nothing uses data after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-7,
    obs_var: float = 1e-3,
    init_state: tuple[float, float] | None = None,
    init_cov: float = 1.0,
) -> pd.DataFrame:
    """Run the filter and return columns beta, alpha, innovation, innovation_std, zscore."""
    if not y.index.equals(x.index):
        raise ValueError("y and x must share the same index")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if obs_var <= 0 or init_cov <= 0:
        raise ValueError("obs_var and init_cov must be positive")

    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    if not (np.isfinite(yv).all() and np.isfinite(xv).all()):
        raise ValueError("inputs must be finite")
    n = len(yv)

    theta = np.array(init_state if init_state is not None else (0.0, 0.0), dtype=float)
    P = np.eye(2) * init_cov
    Q = np.eye(2) * delta / (1.0 - delta)

    beta = np.empty(n)
    alpha = np.empty(n)
    innov = np.empty(n)
    innov_std = np.empty(n)

    for t in range(n):
        if t > 0:
            P = P + Q  # predict step (state transition is identity)
        F = np.array([xv[t], 1.0])
        e = yv[t] - F @ theta
        S = float(F @ P @ F) + obs_var
        K = (P @ F) / S
        theta = theta + K * e
        P = P - np.outer(K, F @ P)
        P = 0.5 * (P + P.T)  # enforce symmetry against round-off
        beta[t], alpha[t] = theta
        innov[t] = e
        innov_std[t] = np.sqrt(S)

    return pd.DataFrame(
        {
            "beta": beta,
            "alpha": alpha,
            "innovation": innov,
            "innovation_std": innov_std,
            "zscore": innov / innov_std,
        },
        index=y.index,
    )
