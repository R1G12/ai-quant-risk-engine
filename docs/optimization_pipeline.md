# Optimization Pipeline

## DVC flow (Phase 3)

```
merge_features
  → generate_volatility_metrics
  → generate_var_metrics → generate_cvar_metrics
  → generate_correlations
  → generate_portfolio_metrics
  → optimize_portfolios
  → generate_efficient_frontier
```

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
dvc repro merge_features generate_volatility_metrics generate_var_metrics
dvc repro generate_cvar_metrics generate_correlations generate_portfolio_metrics
dvc repro optimize_portfolios generate_efficient_frontier
```

Holdings must match active market tickers. If you change `configs/run.yaml` or merge branches with different tickers, run `aqre prepare` or let `ensure_holdings()` refresh stale `holdings.parquet` on the next risk stage.

For `weighting: optimised` or `partial`, the Streamlit Portfolio page reads **`optimal_weights.parquet`**, not the equal placeholder in holdings — see [portfolio_dashboard.md](portfolio_dashboard.md).

**Position signs:** `optimize_portfolios` calls `effective_position_sides()` so explicit `position_sides` plus optional FinBERT inference (±0.3 thresholds) set long/short before max-Sharpe / partial. Requires Phase 1 `sentiment.parquet` for inference. See [run_profile.md — FinBERT position sides](run_profile.md#finbert-position-sides).

Install risk extras: `pip install -e ".[dev,risk]"`
