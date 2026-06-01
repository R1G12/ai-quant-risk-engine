# Optimization Pipeline

## DVC flow (Phase 3)

```
merge_features
  → generate_correlations
  → optimize_portfolios          # writes optimal_weights.parquet first
  → generate_portfolio_metrics   # portfolio returns, regimes, metrics (uses effective weights)
  → generate_volatility_metrics
  → generate_var_metrics → generate_cvar_metrics
  → generate_efficient_frontier
  → … Phase 4 (simulations, run_backtests, …)
```

**Why optimize runs before VaR / portfolio metrics:** `build_portfolio_returns()` uses `load_portfolio_weights()`, which reads `optimal_weights.parquet` when `weighting` is `optimised` or `partial`. Running optimization first keeps **Sharpe KPIs, VaR, drawdown charts, HMM regimes, and MC calibration** aligned with the same weights as the Portfolio page.

**Regime mixer note:** `regime_sentiment_mix` at optimization time reads the **previous** `regimes.parquet` (if any). After `generate_portfolio_metrics`, regimes are refit on the new return series; the **next** full `dvc repro` picks up updated labels.

## Outputs

| Path | Content |
|------|---------|
| `data/risk/volatility/` | Partitioned vol metrics |
| `data/risk/var/` | VaR table |
| `data/risk/cvar/` | CVaR table |
| `data/risk/correlations/` | Cov/corr long format |
| `data/risk/portfolio/` | Returns, metrics, regimes |
| `data/risk/optimization/optimal_weights.parquet` | Per-portfolio weights (`max_sharpe`, `min_variance`, …) — used by dashboard when `weighting` is `optimised` / `partial` |
| `data/risk/optimization/` | Frontier and stats |
| `data/analytics/risk/` | Plotly HTML |

## Run

**Recommended** (applies run profile + holdings):

```powershell
aqre prepare
aqre run profile
```

Or manual DVC (CI / sample):

```powershell
.venv312\Scripts\activate
$env:MARKET_SOURCE = "sample"
$env:RUN_PROFILE = "configs/run.ci.yaml"   # optional; matches CI tickers
dvc repro merge_features generate_correlations optimize_portfolios
dvc repro generate_portfolio_metrics generate_volatility_metrics generate_var_metrics generate_cvar_metrics
dvc repro generate_efficient_frontier run_backtests
```

Holdings must match active market tickers. If you change `configs/run.yaml` or merge branches with different tickers, run `aqre prepare` or let `ensure_holdings()` refresh stale `holdings.parquet` on the next risk stage.

For `weighting: optimised` or `partial`, the Streamlit Portfolio page, VaR, regime charts, and backtest KPIs all use weights from **`optimal_weights.parquet`** after the stages above — see [portfolio_dashboard.md](portfolio_dashboard.md).

**Position sides:** `optimize_portfolios` calls `effective_position_sides()` so explicit `position_sides` plus optional FinBERT inference (±0.3 thresholds) set long/short before max-Sharpe / partial. Requires Phase 1 `sentiment.parquet` for inference. See [run_profile.md — FinBERT position sides](run_profile.md#finbert-position-sides).

**Sentiment μ blend and tilt:** `build_expected_returns()` blends historical μ with 30d FinBERT scores; `regime_sentiment_mix` scales influence using the latest `regimes.parquet` on disk (from the prior pipeline run, if present). Optional magnitude tilt runs after max-Sharpe. See [run_profile.md — Sentiment in optimization](run_profile.md#sentiment-in-optimization).

Install risk extras: `pip install -e ".[dev,risk]"`
