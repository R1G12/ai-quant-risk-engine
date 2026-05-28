# Run profile (`configs/run.yaml`)

Single place to configure a **reproducible pipeline run** (tickers, demo vs live data, portfolio weighting).

The trading notebook [`notebooks/trading_risk_manager_final.ipynb`](../notebooks/trading_risk_manager_final.ipynb) remains **standalone**; copy values from this file manually if you want the notebook to match a pipeline run.

## Quick start

```powershell
aqre prepare
aqre run profile
```

Or one shot:

```powershell
aqre run profile --dashboard
```

## `mode`

| Value | Market data | News ingestion |
|-------|-------------|----------------|
| `demo` | `sample` (local synthetic) | `sample` |
| `live` | `yfinance` | `sample` (real news adapter is Phase 5 backlog) |

`aqre prepare` and `aqre run profile` set `MARKET_SOURCE` from `mode` (ignores a leftover `MARKET_SOURCE=sample` in the shell). Plain `dvc repro` still respects `MARKET_SOURCE` for CI.

## `market`

- **tickers**: universe for ingest, features, risk, and research.
- **Validation**: `aqre prepare` and market ingest **skip** symbols with no data (unknown/delisted). Warnings list skipped tickers; holdings and the manifest only include valid symbols. Sample mode never invents synthetic prices for missing tickers.
- **use_rolling_window** / **rolling_days**: end date = today, start = today − rolling_days (unless `MARKET_PIN_DATES=1`).

## `portfolio`

- **weighting**:
  - `equal` — equal gross weight across tickers (respecting `position_sides`).
  - `manual` — use `manual_weights` (normalized to gross budget 1).
  - `partial` — fix `anchor_weights`, optimize free tickers at Phase 3.
  - `optimised` — max-Sharpe on gross budget with optional shorts.
- **allow_shorts** / **max_gross_per_ticker**: gross-budget constraints (notebook-style).
- **position_sides**: force sign per ticker, e.g. `QQQ: short`.
- **anchor_weights**: used when `weighting: partial`; sum(|anchors|) must be &lt; 1.
- **risk_free**: annual rate for optimization Sharpe.

`aqre prepare` writes `data/raw/portfolio/holdings.parquet` and `data/run_manifest.json`.

## `research`

- **backtest_weight_source**: portfolio label in optimization output (default `max_sharpe`).

## Files produced

| Path | Purpose |
|------|---------|
| `data/raw/portfolio/holdings.parquet` | Static weights for early pipeline stages |
| `data/run_manifest.json` | Run metadata (gitignored) |
| `data/risk/optimization/` | Optimized weights after Phase 3 |
