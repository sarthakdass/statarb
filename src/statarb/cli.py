"""Command-line entry point.

Examples
--------
statarb --source synthetic
statarb --source yahoo --start 2010-01-01 \\
    --pairs KO PEP XOM CVX V MA HD LOW \\
    --alloc SPY EFA EEM TLT GLD --hedge kalman --output results/
statarb --source csv --pairs-csv pairs.csv --alloc-csv alloc.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import ALLOCATION_METHODS, COVARIANCE_METHODS, AllocationConfig, CostModel, PairsConfig
from .data import load_csv, load_yahoo, simulate_prices
from .strategy import run_composite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="statarb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["synthetic", "yahoo", "csv"], default="synthetic")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--pairs", nargs="+", default=["KO", "PEP", "XOM", "CVX", "V", "MA"])
    p.add_argument("--alloc", nargs="+", default=["SPY", "EFA", "EEM", "TLT", "GLD"])
    p.add_argument("--pairs-csv", type=Path)
    p.add_argument("--alloc-csv", type=Path)
    p.add_argument("--seed", type=int, default=7)

    g = p.add_argument_group("pairs")
    g.add_argument("--hedge", choices=["ols", "kalman"], default="ols")
    g.add_argument("--formation", type=int, default=252)
    g.add_argument("--trading", type=int, default=63)
    g.add_argument("--entry", type=float, default=2.0)
    g.add_argument("--exit", type=float, default=0.5)
    g.add_argument("--stop", type=float, default=4.0)
    g.add_argument("--max-pairs", type=int, default=5)
    g.add_argument("--pvalue", type=float, default=0.05)
    g.add_argument("--no-fdr", action="store_true")

    a = p.add_argument_group("allocation")
    a.add_argument("--alloc-method", choices=ALLOCATION_METHODS, default="risk_parity")
    a.add_argument("--cov", choices=COVARIANCE_METHODS, default="ledoit_wolf")
    a.add_argument("--lookback", type=int, default=252)
    a.add_argument("--rebalance", type=int, default=21)
    a.add_argument("--max-weight", type=float, default=1.0)
    a.add_argument("--trend-filter", action="store_true")

    c = p.add_argument_group("portfolio & costs")
    c.add_argument("--tactical-weight", type=float, default=0.3)
    c.add_argument("--commission-bps", type=float, default=1.0)
    c.add_argument("--slippage-bps", type=float, default=2.0)
    c.add_argument("--borrow-bps", type=float, default=50.0)
    c.add_argument("--output", type=Path, default=None, help="directory for CSVs and a chart")
    return p


def _load(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.source == "synthetic":
        prices, _ = simulate_prices(n_days=2500, n_pairs=4, n_independent=5, seed=args.seed)
        alloc_cols = [c for c in prices.columns if c.startswith("IDX")]
        return prices, prices[alloc_cols]
    if args.source == "yahoo":
        return (
            load_yahoo(args.pairs, args.start, args.end),
            load_yahoo(args.alloc, args.start, args.end),
        )
    if args.pairs_csv is None or args.alloc_csv is None:
        raise SystemExit("--source csv requires --pairs-csv and --alloc-csv")
    return load_csv(args.pairs_csv), load_csv(args.alloc_csv)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs_prices, alloc_prices = _load(args)
    pairs_cfg = PairsConfig(
        formation_days=args.formation,
        trading_days=args.trading,
        entry_z=args.entry,
        exit_z=args.exit,
        stop_z=args.stop,
        max_pairs=args.max_pairs,
        pvalue_threshold=args.pvalue,
        fdr_control=not args.no_fdr,
        hedge_method=args.hedge,
    )
    alloc_cfg = AllocationConfig(
        method=args.alloc_method,
        covariance=args.cov,
        lookback_days=args.lookback,
        rebalance_every=args.rebalance,
        max_weight=args.max_weight,
        trend_filter=args.trend_filter,
    )
    costs = CostModel(args.commission_bps, args.slippage_bps, args.borrow_bps)
    res = run_composite(
        pairs_prices,
        alloc_prices,
        pairs_cfg,
        alloc_cfg,
        tactical_weight=args.tactical_weight,
        costs=costs,
    )

    ew_bench = alloc_prices.pct_change().mean(axis=1)  # daily-rebalanced equal weight
    table = res.summary_table(benchmark=ew_bench)
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 120):
        print(table.to_string())
        n_sel = len(res.pairs.selections)
        print(f"\nPair selections across windows: {n_sel}")
        if n_sel:
            counts = res.pairs.selections.groupby(["y", "x"]).size().sort_values(ascending=False)
            print(counts.head(10).to_string())

    if args.output is not None:
        out = args.output
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "tactical": res.tactical.returns,
                "strategic": res.strategic.returns,
                "combined": res.combined.returns,
            }
        ).to_csv(out / "daily_returns.csv")
        table.to_csv(out / "summary.csv")
        res.pairs.selections.to_csv(out / "pair_selections.csv", index=False)
        res.strategic.weights.to_csv(out / "allocation_weights.csv")
        try:
            from .plotting import plot_equity

            plot_equity(
                {
                    "tactical": res.tactical.returns,
                    "strategic": res.strategic.returns,
                    "combined": res.combined.returns,
                },
                out / "equity.png",
            )
        except ImportError:
            print("matplotlib not installed; skipping chart", file=sys.stderr)
        print(f"\nWrote results to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
