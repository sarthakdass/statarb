# statarb

A walk-forward research framework for two complementary quant strategies:

- **s-arbitrage (pairs trading).** Cointegration-based
  pair selection with false-discovery-rate control, static-OLS or Kalman-filter
  hedge ratios, z-score entry/exit/stop/time-stop rules, and rolling
  formation → trading windows so every decision uses only past data.
- **Strategic sleeve — asset allocation.** Rolling equal-weight, inverse-volatility,
  minimum-variance, maximum-Sharpe and equal-risk-contribution (risk parity)
  portfolios, with sample / Ledoit-Wolf / EWMA covariance, mean shrinkage, weight
  caps and an optional trend filter.
- **One backtest engine** for both: NAV-relative accounting with weight drift,
  execution lag, commissions + slippage on turnover, and borrow cost on shorts.
  Sleeves are combined into a single book with periodic rebalancing.

See [`docs/THEORY.md`](docs/THEORY.md) for the methodology, and for a
critique of common pitfalls (look-ahead, correlation-vs-cointegration,
overlapping-return Sharpe ratios, in-sample optimisation).

## Install

```bash
git clone <your-repo-url> statarb && cd statarb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data,plot]"
pytest
```

Python ≥ 3.10. Core dependencies: numpy, pandas, scipy, statsmodels.
`yfinance` (real data) and `matplotlib` (charts) are optional extras.

## Quick start

```bash
# synthetic market with known cointegrated pairs (no network required)
statarb --source synthetic

# real data
statarb --source yahoo --start 2010-01-01 \
    --pairs KO PEP XOM CVX HD LOW V MA \
    --alloc SPY EFA EEM TLT GLD \
    --hedge kalman --alloc-method risk_parity --output results/

# your own data: wide CSVs (date column, then one price column per ticker)
statarb --source csv --pairs-csv pairs.csv --alloc-csv alloc.csv
```

```python
from statarb import (AllocationConfig, CostModel, PairsConfig,
                     load_yahoo, run_composite)

pairs_px = load_yahoo(["KO", "PEP", "XOM", "CVX", "HD", "LOW"], "2010-01-01")
alloc_px = load_yahoo(["SPY", "EFA", "EEM", "TLT", "GLD"], "2010-01-01")

res = run_composite(
    pairs_px, alloc_px,
    PairsConfig(hedge_method="kalman", entry_z=2.0, exit_z=0.5, stop_z=4.0),
    AllocationConfig(method="risk_parity", covariance="ledoit_wolf"),
    tactical_weight=0.3,
    costs=CostModel(commission_bps=1, slippage_bps=2, borrow_bps_annual=50),
)
print(res.summary_table())
```

`examples/research_workflow.py` compares hedge models and allocation rules side by side.

## Layout

```
src/statarb/
  config.py      validated dataclasses: PairsConfig, AllocationConfig, CostModel
  data.py        cleaning/alignment, CSV + yfinance loaders, synthetic market simulator
  stats.py       OLS hedge ratio, Engle-Granger, AR(1) half-life, Benjamini-Hochberg
  kalman.py      time-varying hedge ratio with innovation z-scores
  pairs.py       pair screening, position state machine, walk-forward targets
  covariance.py  sample, Ledoit-Wolf (matches scikit-learn), EWMA
  optimize.py    inverse-vol, min-variance, max-Sharpe (multi-start), risk parity (Spinu)
  allocation.py  rolling allocation targets, sleeve combination
  backtest.py    drift/lag/cost-aware engine
  metrics.py     CAGR, vol, Sharpe, Sortino, drawdown, beta/alpha, probabilistic Sharpe
  strategy.py    high-level runners
  cli.py         command-line interface
tests/           60 tests incl. no-look-ahead checks (future data perturbed → past unchanged)
```

## Conventions

- A target dated *t* uses prices up to the close of *t*; the engine executes it at
  the close of *t + execution_lag* (default 1).
- Weights are fractions of NAV; `cash = 1 − Σw` (short proceeds sit in cash).
- Returns are compounded (`cumprod`), never summed.
- The pairs sleeve allocates `gross_leverage / max_pairs` per pair. Its
  unlevered volatility is low, so if you combine it with a long-only sleeve,
  size it by risk (raise `gross_leverage`) rather than by capital.

## Limitations

- Daily close-to-close simulation: no intraday fills, market impact is linear, and
  short availability is assumed.
- The asset universe is whatever you pass in. Picking pairs you *already know* stayed
  related (e.g. KO/PEP today) is survivorship/selection bias; use a point-in-time
  universe for serious research.
- Engle-Granger on a 1-year window has low power for half-lives much above ~10
  days; FDR control makes selection conservative by design.
- `load_yahoo` depends on yfinance and Yahoo's terms of use; use a licensed vendor
  for anything beyond personal research.

