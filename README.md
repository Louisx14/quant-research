# Quant Research — Systematic Strategy Studies

Research studies behind ~18 months of independent systematic trading work on crypto market data
(SQLite candle databases built with the companion [market-data-pipeline](../market-data-pipeline) repo),
carried out under the mentorship of a professional quantitative PM.

These are research scripts, not a package: each folder is a study, numbered files show the
progression of the work. Code is kept close to its original working state on purpose —
the point is the methodology and the results, including the negative ones.

## Studies

### `conditional_density/` — kernel conditional density estimation
Estimating the distribution of forward returns *conditional* on observable state, using
quantile-binned conditional histograms and smoothed kernel density estimates. Includes
fee-adjusted expected-value comparison, conditional bull-probability estimation, and an
ACF check on the conditioning variables.

### `synthetic_markets/` — block-bootstrap synthetic data
Generation of synthetic OHLC series (random-draw and block-bootstrap models) used as a
null benchmark: a strategy has to beat its performance distribution on synthetic markets,
not just look good on one historical path.

### `trend_following/` — regression-ensemble trend following (REGEX)
Regression-based trend detection over multiple lookback parameters, with an empirical
slippage-estimation study (`regex_03`) measuring close-to-next-open execution gaps against
tolerance assumptions. One deployment of this family went live (see pipeline repo).

### `correlation_drawdown/` — cross-asset correlation as a drawdown predictor
Tested a novel aggregate cross-asset correlation measure as a forward drawdown predictor
over 18 years of equity data. **Result: null**, documented across 3 specifications including
a random-sampling hypothesis test (`cor_05`). Kept public deliberately — a documented null
with a proper test beats an undocumented "win".

### `volatility/` — volatility structure studies
Stablecoin-volume/volatility correlation and volatility-over-time structure used as
regime context for the strategies above.

## Honest limitations
- Scripts assume a local SQLite candle DB at `data/` (see pipeline repo to build one)
- Research code: hardcoded parameters, some magic column indices, minimal error handling
- One live strategy from this research was retired after live-vs-backtest reconciliation
  uncovered a look-ahead bias that turned per-trade EV negative — the reconciliation
  tooling lives in the pipeline repo (`execution/trade_reconstruction.py`)
