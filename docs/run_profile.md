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
| `live` | `yfinance` | `yfinance` |

`aqre prepare` and `aqre run profile` set `MARKET_SOURCE` and `NEWS_SOURCE` from `mode` (ignores leftover shell overrides when using `--force` / profile commands). Plain `dvc repro` still respects `MARKET_SOURCE` and `NEWS_SOURCE` for CI.

## `market`

- **tickers**: universe for ingest, features, risk, and research.
- **Validation**: `aqre prepare` and market ingest **skip** symbols with no data (unknown/delisted). Warnings list skipped tickers; holdings and the manifest only include valid symbols. Sample mode never invents synthetic prices for missing tickers.

**CI** uses [`configs/run.ci.yaml`](../configs/run.ci.yaml) (`RUN_PROFILE` / `CI=true`) with sample-compatible tickers; local runs use [`configs/run.yaml`](../configs/run.yaml).
- **use_rolling_window** / **rolling_days**: end date = today, start = today − rolling_days (unless `MARKET_PIN_DATES=1`).

## `portfolio`

- **weighting**:
  - `equal` — equal gross weight across tickers (respecting `position_sides`).
  - `manual` — use `manual_weights` (normalized to gross budget 1).
  - `partial` — fix `anchor_weights`, optimize free tickers at Phase 3.
  - `optimised` — max-Sharpe on gross budget with optional shorts.
- **allow_shorts** / **max_gross_per_ticker**: gross-budget constraints (notebook-style).
- **position_sides**: force sign per ticker, e.g. `QQQ: short` (overrides FinBERT inference).
- **min_gross_divisor** (default `5`): minimum gross per name is `1 / (divisor × n_tickers)` during optimization.
- **anchor_weights**: used when `weighting: partial`; sum(|anchors|) must be &lt; 1.
- **risk_free**: annual rate for optimization Sharpe.

## FinBERT position sides

- **sentiment_position_sides** (default `true`): for tickers **not** in `position_sides`, set `short` when 30d FinBERT score ≤ −0.3 and `long` when ≥ +0.3 (same bands as trailing stops). Requires `allow_shorts: true` and `data/processed/sentiment.parquet`.
- **sentiment_sides_window_days** (default `30`): lookback for inference (matches Signals dashboard).

## Sentiment in optimization

Blended expected returns and optional tilt use the same FinBERT scores as Signals:

- **sentiment_mu_blend** (default `0.3`): weight α on sentiment-based μ vs historical μ.
- **sentiment_mu_mode**: `vol_scaled` (scale by |historical return|) or `fixed`.
- **sentiment_mu_scale**: scale factor for sentiment μ leg.
- **regime_sentiment_mix**: multipliers by latest HMM label (`low` / `mid` / `high`) on α — see `data/risk/portfolio/regimes.parquet`.
- **sentiment_magnitude_tilt** (default `true`): post max-Sharpe z-score tilt on weight magnitudes (`sentiment_tilt_beta`, `sentiment_tilt_cap`).
- **use_legacy_bullish_mu**: if `true`, use old `bullish_ratio` nudge instead of blend.

Optimization metadata (`data/risk/optimization/_metadata.json`) records `mu_mode`, `regime`, `alpha_eff`, `tilt_applied`.

**CI** ([`configs/run.ci.yaml`](../configs/run.ci.yaml)) sets `sentiment_position_sides: false`, `sentiment_mu_blend: 0`, `sentiment_magnitude_tilt: false` for reproducibility.

`aqre prepare` writes `data/raw/portfolio/holdings.parquet` and `data/run_manifest.json`.

**Stale holdings:** if `holdings.parquet` exists but tickers no longer match the active profile (e.g. after merging `main` or switching `run.yaml` ↔ `run.ci.yaml`), risk stages call `ensure_holdings()` and rewrite equal weights for the current universe. Custom weights from a prior `aqre prepare` are kept when file tickers still match.

## `research`

- **backtest_weight_source**: portfolio label in optimization output (default `max_sharpe`).

## Files produced

| Path | Purpose |
|------|---------|
| `data/raw/portfolio/holdings.parquet` | Static weights for early pipeline stages |
| `data/run_manifest.json` | Run metadata (gitignored) |
| `data/risk/optimization/` | Optimized weights after Phase 3 |
