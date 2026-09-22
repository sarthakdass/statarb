# Methodology

This document explains the two strategies implemented in `statarb`, motivated 
by ("Quant Trading 101: Pairs & Allocations", Onepagecode, 2025).

---

## Part 1 — Pairs trading

### 1.1 Economic premise

Two assets exposed to the same fundamentals (Coca-Cola and PepsiCo; two
integrated oil majors; two card networks) should respond similarly to shared
news. If some linear combination of their prices is *stationary*, deviations
from that combination are temporary and can be traded: buy the cheap leg, 
short the rich leg, and close when the gap closes. Because one leg is long 
and the other short, most market exposure cancels, so the P&L comes from 
relative moves rather than market direction.

### 1.2 Correlation better than cointegration

Correlation measures co-movement of *returns*. Two stocks can have highly
correlated daily returns while their price levels drift apart forever — nothing
pulls them back. What a pairs trade needs is *mean reversion of the price gap*,
which is the definition of cointegration: two I(1) (random-walk-like) log-price
series $y_t, x_t$ are cointegrated if there is a $\beta$ such that

$$s_t = y_t - \alpha - \beta x_t$$

is I(0), i.e. stationary.

**Engle–Granger two-step test** (`stats.engle_granger`):

1. Regress $y_t$ on $x_t$ by OLS to estimate $\hat\alpha,\hat\beta$.
2. Run an augmented Dickey–Fuller test on the residual $\hat s_t$. Because the
   residual was fitted to look stationary, the usual ADF critical values are
   wrong; MacKinnon's cointegration critical values are used (statsmodels
   `coint`).

The test is asymmetric in $(y, x)$; the code fixes a convention (alphabetical
order, first regressed on second). Johansen's test is the symmetric
multivariate alternative.

**Power.** Over a 252-day window, a spread with half-life $h$ has AR(1)
coefficient $\phi = 2^{-1/h}$. The expected ADF t-statistic is roughly
$(\phi-1)/\sqrt{(1-\phi^2)/T}$; for $h=16$ this is about $-2.3$ against a
5% critical value near $-3.4$. One year of daily data simply cannot confirm
slow mean reversion. Longer formation windows help but raise the risk that the
relationship has changed.

### 1.3 Multiple testing

Screening $N$ tickers means testing $N(N-1)/2$ pairs. With 50 tickers that is
1,225 tests; at a 5% threshold ~61 pairs "pass" even if none is cointegrated.
`find_pairs` applies the **Benjamini–Hochberg** procedure: sort the p-values,
find the largest $k$ with $p_{(k)} \le qk/m$, and reject the $k$ smallest.
This bounds the expected fraction of false discoveries among selected pairs
at $q$. It is deliberately conservative.

### 1.4 Ornstein–Uhlenbeck process spread; half-life

A stationary spread is modelled as a discretised OU process
$s_t = c + \phi s_{t-1} + \varepsilon_t$ with $0<\phi<1$. The half-life
(time for a deviation to decay by half) is

$$h = -\ln 2 / \ln \phi.$$

Half-life filters out pairs that revert too fast to trade after costs
(< 2 days) or too slowly to be useful within a trading window (> 60 days),
and it sets the **time stop**: if a trade has not converged after
$k \cdot h$ days, the relationship is probably broken.

### 1.5 Hedge ratio: static OLS vs Kalman filter

*Static OLS* estimates $\beta$ once per formation window and holds it through
the trading window. Simple, but relationships drift.

*Kalman filter* (`kalman.py`) treats $(\beta_t,\alpha_t)$ as a random walk:

- observation $y_t = \beta_t x_t + \alpha_t + e_t,\ e_t\sim N(0,R)$
- state $\theta_t = \theta_{t-1} + w_t,\ w_t \sim N(0, \tfrac{\delta}{1-\delta}I)$

Each day the filter forecasts $y_t$ with yesterday's state, and the forecast
error $e_t$ divided by its predicted standard deviation $\sqrt{S_t}$ is a
z-score. $\delta$ controls how fast the hedge adapts. **Scaling matters:** on
log prices ($x\approx 3$–$6$), per-step state noise is about $x^2\delta$. The
often-quoted $\delta=10^{-4}$ (calibrated for price *levels* in Chan's
examples) makes this ~$10^{-3}$, larger than the spread variance, so the filter
absorbs the spread into $\beta$ and $\alpha$ and z-scores never reach entry
levels. The default here is $10^{-7}$; the test suite checks that z-scores
are approximately unit-variance.

### 1.6 From spread to trades

z-score: $z_t = (s_t - \mu)/\sigma$, with $\mu, \sigma$ from the formation
window (static), a rolling window, or the Kalman innovation variance.

Position rules (`generate_positions`), all using $z_t$ only:

| condition | action |
|---|---|
| flat and $z \ge +\text{entry}$ | short spread (short $y$, long $\beta x$) |
| flat and $z \le -\text{entry}$ | long spread |
| long and $z \ge -\text{exit}$ (or short and $z \le \text{exit}$) | take profit |
| $\lvert z\rvert \ge \text{stop}$ | stop-loss; disarm until $\lvert z\rvert < \text{entry}$ |
| held $\ge k\cdot h$ days | time stop; disarm |

Disarming prevents the classic bug of re-entering, on the next bar, a spread
that just blew through the stop.

**Sizing.** For the log spread, $\Delta s = r_y - \beta r_x$, so dollar weights
are proportional to $(1, -\beta)$; they are scaled to a fixed gross exposure
per pair. This is beta-weighted, not dollar-neutral, unless $\beta=1$.

### 1.7 Walk-forward protocol

```
|---- formation (select, estimate) ----|-- trade --|
             |---- formation ----|-- trade --|
```

Pairs, hedge ratios, spread mean/std and half-lives are all estimated on the
formation window only, then frozen (or updated causally by the Kalman filter)
through the following trading window. The test suite enforces this: it
perturbs all prices after a cut-off date and asserts that every target before
the cut-off is bit-for-bit unchanged.

### 1.8 Criticisms and changes from onepagecode

The tutorial's KO/PEP strategy buys whichever stock fell more yesterday when a
9-day rolling correlation of overlapping 19-day returns drops below 0.9. Issues:

- **It is not a pairs trade.** It is long-only, so it carries full market beta and 
  there is no short leg and no hedge ratio.
- **Correlation, not cointegration**, drives the signal; overlapping 19-day
  returns make the 9-point correlation extremely noisy.
- **`Target.isin(pairs['TPEP'])` compares against the whole column**, not the
  same row. Whenever both stocks' returns are equal (e.g. both exactly 0.0 on
  an unchanged close), the classification is ambiguous and defaults to PEP.
  A row-wise comparison (`pairs['TPEP'] <= pairs['TKO']`) is correct.
- **Returns are summed (`cumsum`)**, not compounded, and the printed "Sharpe"
  divides the final cumulative return by the standard deviation of the
  cumulative-return path — not a Sharpe ratio. The printed "beta" is a
  correlation.
- **No transaction costs**, on a strategy that trades almost daily.
- **Parameters were chosen on the same data used to report performance**; the
  author notes the "alpha decay from 2011 onwards" — that is what in-sample
  fitting looks like out of sample.
- **Selection bias**: KO and PEP were picked with hindsight as a pair that
  stayed related.

---

## Part 2 — Asset allocation

### 2.1 The problem

Choose weights $w$ over $N$ assets to trade off expected return
$\mu^\top w$ against risk $w^\top \Sigma w$, subject to constraints (long-only,
fully invested, position caps). Everything hinges on estimates of $\mu$ and
$\Sigma$, and those estimates are noisy.

### 2.2 Estimating $\Sigma$

- *Sample covariance*: unbiased but noisy; with $T$ close to $N$ it becomes
  ill-conditioned, and optimisers exploit its errors.
- *Ledoit–Wolf shrinkage* (`covariance.ledoit_wolf`):
  $\hat\Sigma = (1-\lambda) S + \lambda \bar\sigma^2 I$ with the
  analytically optimal $\lambda$. It is always well conditioned. The
  implementation matches scikit-learn to machine precision (tested).
- *EWMA*: weights recent data more, adapting to volatility regimes.

### 2.3 Estimating $\mu$ (hard)

Mean returns are estimated with standard error $\sigma/\sqrt{T_{\text{years}}}$:
with 16% volatility and 15 years of data, that is ±4% a year, about the size
of the equity premium itself. Mean-variance optimisers are extremely sensitive
to $\mu$ (Michaud's "error maximization"). Mitigations used here: shrink $\mu$
toward the cross-sectional average (`mean_shrinkage`), cap weights, re-estimate
out-of-sample — or avoid $\mu$ entirely.

### 2.4 Construction rules (`optimize.py`)

| rule | needs | objective |
|---|---|---|
| equal weight | nothing | $w_i = 1/N$ |
| inverse volatility | $\sigma_i$ | $w_i \propto 1/\sigma_i$ |
| minimum variance | $\Sigma$ | $\min w^\top\Sigma w$ |
| maximum Sharpe | $\mu,\Sigma$ | $\max (w^\top\mu - r_f)/\sqrt{w^\top\Sigma w}$ |
| risk parity (ERC) | $\Sigma$ | $w_i(\Sigma w)_i$ equal across $i$ |

Maximum Sharpe is non-convex under constraints, so it uses multi-start SLSQP
with analytic gradients; with no asset above $r_f$ it falls back to minimum
variance. Risk parity is solved through Spinu's convex reformulation
$\min_y \tfrac12 y^\top\Sigma y - \sum_i b_i\ln y_i$, which has a unique
solution; normalising $y$ gives risk contributions exactly equal to the
budgets $b$.

### 2.5 Rebalancing and drift

Between rebalances, weights drift with prices:
$w_i \leftarrow w_i(1+r_i)/(1+r_p)$. The engine simulates this and charges
costs only on the turnover needed to return to target. Rebalancing frequency
trades tracking error against costs.

### 2.6 Criticisms and changes from onepagecode

- **Overlapping returns.** Rolling 252-day returns sampled daily overlap by
  251 days; there are only about $T/252$ independent observations (~15 over 15
  years). A "Sharpe" computed as mean/std of that series is not a Sharpe ratio
  and cannot be compared to one computed on daily returns.
- **In-sample optimization.** The max-Sharpe weights (92.5% S&P 500) were
  fitted on the full history and then evaluated on the same history — a
  guaranteed "win" that says little about the future. It mostly encodes that
  US equities happened to outperform over 2004–2020.
- **Hard-coded regime thresholds** (−0.17 on the S&P 1-year return, 0.29 on
  the SSE) were tuned in-sample; the dynamic strategy's outperformance is
  largely that fit.
- **Currency.** The TSX, STOXX 600 and SSE are in CAD, EUR and CNY. Mixing them
  unhedged without conversion means the "returns" of a portfolio are not
  returns of any investable portfolio in one currency.
- **Non-smooth objective.** Maximising the *median* with a gradient-based
  solver (SLSQP) is unreliable; the median is piecewise constant in $w$.
- **The constraint `1 − Σw ≥ 0`** permits holding cash

This package replaces all of these with walk-forward estimation (weights at
$t$ use only data up to $t$, executed at $t+1$), daily compounded returns,
annualised metrics on non-overlapping daily data, and costs.

---

## Part 3 — Combining methods

The pairs sleeve earns small, low-beta returns; the allocation sleeve earns the
risk premia of its assets. Because their returns are nearly uncorrelated,
combining them can improve risk-adjusted returns. `combine_sleeves` treats each
sleeve's net equity curve as an asset and rebalances between them.

Size sleeves by **risk**, not capital: a pairs sleeve with 1% volatility given
30% of capital contributes almost nothing. Raise its `gross_leverage` (subject
to margin and borrow constraints) or target a volatility budget.

## Part 4 — Evaluation

- Report compounded annual return, annualised volatility, Sharpe on daily
  excess returns, maximum drawdown, turnover and cost drag.
- The **probabilistic Sharpe ratio** (Bailey & López de Prado) gives the
  probability that the true Sharpe exceeds a benchmark, adjusting for sample
  length, skewness and fat tails. The pairs sleeve's large positive skew and
  excess kurtosis are typical; they matter for this statistic.
- Every parameter you tried counts. That is, if you test 100 configurations, 
  the best in-sample Sharpe is inflated; hold out a final period you never
  look at until the end.
