import numpy as np
import pandas as pd
import pytest

from statarb.config import PairsConfig
from statarb.pairs import find_pairs, generate_positions, spread_to_weights, walk_forward_pairs


def test_generate_positions_basic_cycle():
    z = np.array([0.0, 2.1, 1.5, 0.4, 0.0, -2.5, -1.0, -0.3, 0.0])
    pos = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    np.testing.assert_array_equal(pos, [0, -1, -1, 0, 0, 1, 1, 0, 0])


def test_stop_loss_disarms_until_back_inside_entry():
    z = np.array([2.2, 3.0, 4.5, 3.5, 2.5, 1.0, 2.3, 0.2])
    pos = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    # enter short at 2.2, stop at 4.5, no re-entry at 3.5/2.5, re-arm at 1.0, re-enter at 2.3
    np.testing.assert_array_equal(pos, [-1, -1, 0, 0, 0, 0, -1, 0])


def test_no_entry_beyond_stop_and_time_stop():
    z = np.array([5.0, 2.5, 2.4, 2.3, 2.2, 2.1, 1.0, 2.5])
    pos = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0, max_hold=2)
    np.testing.assert_array_equal(pos, [0, -1, -1, 0, 0, 0, 0, -1])


def test_nan_flattens():
    pos = generate_positions(np.array([2.5, np.nan, 2.5]), 2.0, 0.5, 4.0)
    np.testing.assert_array_equal(pos, [-1, 0, -1])


def test_spread_weights_gross_and_hedge():
    w_y, w_x = spread_to_weights(np.array([1.0, -1.0, 0.0]), np.array([0.5, 2.0, 1.0]), 0.2)
    np.testing.assert_allclose(np.abs(w_y) + np.abs(w_x), [0.2, 0.2, 0.0])
    np.testing.assert_allclose(w_x / np.where(w_y == 0, 1, w_y), [-0.5, -2.0, 0.0])


def test_find_pairs_recovers_true_pairs(market):
    prices, truth = market
    window = prices.iloc[:400]
    raw = {(c.y, c.x) for c in find_pairs(window, PairsConfig(max_pairs=20, fdr_control=False))}
    assert {(t.y, t.x) for t in truth} <= raw
    # FDR control can only remove discoveries, never add them
    fdr = {(c.y, c.x) for c in find_pairs(window, PairsConfig(max_pairs=20))}
    assert fdr <= raw
    # on a longer window the test has enough power to survive FDR control
    long = {(c.y, c.x) for c in find_pairs(prices, PairsConfig(max_pairs=20))}
    assert {(t.y, t.x) for t in truth} <= long


def test_find_pairs_rejects_noise():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 6)), axis=0)),
        index=idx,
        columns=list("ABCDEF"),
    )
    assert len(find_pairs(prices, PairsConfig())) <= 1  # FDR keeps false discoveries rare


@pytest.mark.parametrize("hedge,lookback", [("ols", None), ("ols", 60), ("kalman", None)])
def test_walk_forward_has_no_lookahead(market, hedge, lookback):
    prices, _ = market
    cfg = PairsConfig(
        formation_days=200, trading_days=50, hedge_method=hedge, zscore_lookback=lookback
    )
    cut = 520
    full = walk_forward_pairs(prices, cfg).targets
    rng = np.random.default_rng(99)
    future = prices.copy()
    shock = np.exp(np.cumsum(rng.normal(0, 0.05, (len(prices) - cut, prices.shape[1])), axis=0))
    future.iloc[cut:] = prices.iloc[cut:].to_numpy() * shock
    perturbed = walk_forward_pairs(future, cfg).targets
    pd.testing.assert_frame_equal(full.iloc[:cut], perturbed.iloc[:cut])
    assert not full.iloc[:cut].eq(0).all().all()  # the test is not vacuous


def test_walk_forward_respects_gross_limit(market):
    prices, _ = market
    cfg = PairsConfig(formation_days=200, trading_days=50, max_pairs=3, gross_leverage=1.5)
    res = walk_forward_pairs(prices, cfg)
    assert res.targets.abs().sum(axis=1).max() <= 1.5 + 1e-12
    assert res.targets.iloc[:200].eq(0).all().all()
    assert not res.selections.empty


def test_config_validation():
    with pytest.raises(ValueError):
        PairsConfig(entry_z=1.0, exit_z=1.5)
    with pytest.raises(ValueError):
        PairsConfig(zscore_lookback=500, formation_days=252)
