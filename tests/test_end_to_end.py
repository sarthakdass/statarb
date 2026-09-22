import numpy as np

from statarb.config import AllocationConfig, CostModel, PairsConfig
from statarb.strategy import run_composite, run_pairs_strategy


def test_pairs_strategy_trades_true_pairs_and_is_market_neutral(market):
    prices, truth = market
    bt, pr = run_pairs_strategy(
        prices, PairsConfig(formation_days=200, trading_days=50), CostModel()
    )
    pairs = set(zip(pr.selections["y"], pr.selections["x"], strict=True))
    assert any((t.y, t.x) in pairs for t in truth)
    assert bt.turnover.sum() > 0
    mkt = prices.pct_change().mean(axis=1)
    s = bt.summary(benchmark=mkt)
    assert abs(s["beta"]) < 0.1


def test_composite_runs(market):
    prices, _ = market
    alloc = prices[[c for c in prices.columns if c.startswith("IDX")]]
    res = run_composite(
        prices,
        alloc,
        PairsConfig(formation_days=200, trading_days=50),
        AllocationConfig(lookback_days=120),
        tactical_weight=0.4,
    )
    table = res.summary_table()
    assert list(table.columns) == ["tactical_pairs", "strategic_allocation", "combined"]
    assert np.isfinite(res.combined.returns).all()


def test_cli_smoke(capsys):
    from statarb.cli import main

    assert main(["--source", "synthetic", "--seed", "3"]) == 0
    assert "combined" in capsys.readouterr().out
