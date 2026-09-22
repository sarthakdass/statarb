"""End-to-end research workflow.

Run on synthetic data (no network needed):
    python examples/research_workflow.py

Run on real data (requires `pip install -e ".[data,plot]"`):
    python examples/research_workflow.py --yahoo
"""

from __future__ import annotations

import argparse

import pandas as pd

from statarb import (
    AllocationConfig,
    CostModel,
    PairsConfig,
    load_yahoo,
    run_allocation,
    run_composite,
    run_pairs_strategy,
    simulate_prices,
)

PAIRS_UNIVERSE = ["KO", "PEP", "XOM", "CVX", "HD", "LOW", "V", "MA", "UPS", "FDX"]
ALLOC_UNIVERSE = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yahoo", action="store_true")
    args = ap.parse_args()

    if args.yahoo:
        pairs_px = load_yahoo(PAIRS_UNIVERSE, "2008-01-01")
        alloc_px = load_yahoo(ALLOC_UNIVERSE, "2008-01-01")
    else:
        px, truth = simulate_prices(n_days=2500, n_pairs=4, n_independent=5, seed=7)
        pairs_px = px
        alloc_px = px[[c for c in px.columns if c.startswith("IDX")]]
        print("Simulated cointegrated pairs:", [(t.y, t.x) for t in truth])

    costs = CostModel(commission_bps=1, slippage_bps=2, borrow_bps_annual=50)
    fmt = {
        "display.float_format": "{:,.3f}".format,
        "display.width": 140,
        "display.max_columns": None,
    }

    # 1) Tactical sleeve: compare hedge-ratio models, same data and costs.
    rows = {}
    for label, cfg in {
        "ols_static_z": PairsConfig(hedge_method="ols"),
        "ols_rolling_z": PairsConfig(hedge_method="ols", zscore_lookback=60),
        "kalman": PairsConfig(hedge_method="kalman"),
    }.items():
        bt, _ = run_pairs_strategy(pairs_px, cfg, costs)
        rows[label] = bt.summary()
    with pd.option_context(*sum(fmt.items(), ())):
        print(
            "\nPairs sleeve variants\n",
            pd.DataFrame(rows).loc[
                [
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                    "psr_vs_0",
                    "ann_turnover",
                    "ann_cost_drag",
                    "avg_gross_exposure",
                ]
            ],
        )

    # 2) Strategic sleeve: compare allocation rules out of sample.
    rows = {}
    for method in ["equal", "inverse_vol", "min_variance", "max_sharpe", "risk_parity"]:
        bt = run_allocation(alloc_px, AllocationConfig(method=method, max_weight=0.5), costs)
        rows[method] = bt.summary()
    with pd.option_context(*sum(fmt.items(), ())):
        print(
            "\nAllocation rules (walk-forward, net of costs)\n",
            pd.DataFrame(rows).loc[
                ["ann_return", "ann_vol", "sharpe", "max_drawdown", "ann_turnover"]
            ],
        )

    # 3) Combined book.
    res = run_composite(
        pairs_px, alloc_px, PairsConfig(), AllocationConfig(), tactical_weight=0.3, costs=costs
    )
    with pd.option_context(*sum(fmt.items(), ())):
        print("\nComposite\n", res.summary_table())

    try:
        from statarb.plotting import plot_equity

        plot_equity(
            {
                "tactical": res.tactical.returns,
                "strategic": res.strategic.returns,
                "combined": res.combined.returns,
            },
            "equity.png",
            "statarb composite",
        )
        print("\nSaved equity.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
