import numpy as np
import pandas as pd
import pytest

from statarb.kalman import kalman_hedge


def _pair(n=1500, beta=1.3, seed=0):
    rng = np.random.default_rng(seed)
    x = 4 + np.cumsum(rng.normal(0, 0.01, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.85 * s[t - 1] + rng.normal(0, 0.01)
    y = 0.2 + beta * x + s
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(y, idx), pd.Series(x, idx), s.std()


def test_kalman_tracks_constant_beta():
    y, x, sd = _pair()
    kf = kalman_hedge(y, x, delta=1e-7, obs_var=sd**2, init_state=(1.0, 0.0))
    assert kf["beta"].iloc[-200:].mean() == pytest.approx(1.3, abs=0.1)


def test_kalman_zscore_is_well_scaled_with_default_delta():
    # Regression test: too large a delta makes z-scores collapse and the strategy never trades.
    y, x, sd = _pair(seed=4)
    kf = kalman_hedge(y, x, obs_var=sd**2, init_state=(1.3, 0.2))
    z = kf["zscore"].iloc[300:]
    assert 0.6 < z.std() < 1.6
    assert (z.abs() > 2).mean() > 0.01


def test_kalman_is_causal():
    y, x, sd = _pair(n=400)
    base = kalman_hedge(y, x, obs_var=sd**2)
    y2 = y.copy()
    y2.iloc[250:] += 5.0
    pert = kalman_hedge(y2, x, obs_var=sd**2)
    pd.testing.assert_frame_equal(base.iloc[:250], pert.iloc[:250])


def test_kalman_validates_inputs():
    y, x, _ = _pair(n=50)
    with pytest.raises(ValueError):
        kalman_hedge(y, x.iloc[1:])
    with pytest.raises(ValueError):
        kalman_hedge(y, x, delta=0.0)
