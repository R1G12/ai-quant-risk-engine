# Run profile (`configs/run.yaml`)

Single place to configure a **reproducible pipeline run** (tickers, demo vs live data, portfolio weighting).

The trading notebook [`notebooks/trading_risk_manager_final.ipynb`](../notebooks/trading_risk_manager_final.ipynb) remains **standalone**; copy values from this file manually if you want the notebook to match a pipeline run.

## Quick start

```powershell
aqre prepare
aqre run profile                # full dvc repro from configs/run.yaml (no browser)
```

One shot (repro + UI):

```powershell
aqre run profile --dashboard    # same repro, then Phase 5 Streamlit
```

To **only** open the dashboard on data you already built:

```powershell
aqre dashboard
```

`--dashboard` is not “UI without repro” — it is **repro first, UI second**. Use `--skip-repro` with `--dashboard` if you only refreshed holdings and want the app without a full pipeline run.

## `mode`

| Value | Market data | News ingestion |
|-------|-------------|----------------|
| `demo` | `sample` (local synthetic) | `sample` (3 fixed headlines, offline) |
| `live` | `yfinance` | `yfinance` (per-ticker headlines from Yahoo) |

`aqre prepare` and `aqre run profile` set `MARKET_SOURCE` and `NEWS_SOURCE` from `mode` (ignores leftover `MARKET_SOURCE=sample` / `NEWS_SOURCE=sample` in the shell when using `aqre run profile`). Plain `dvc repro` still respects env vars for CI.

Live news requires `pip install -e ".[market]"` and network access for `dvc repro ingest`. Cap headlines per ticker via `ingestion.max_headlines_per_ticker` in [`params.yaml`](../params.yaml) (default `10`).

## `market`

- **tickers**: universe for ingest, features, risk, and research.
- **Validation**: `aqre prepare` and market ingest **skip** symbols with no data (unknown/delisted). Warnings list skipped tickers; holdings and the manifest only include valid symbols. Sample mode never invents synthetic prices for missing tickers.

**CI** uses [`configs/run.ci.yaml`](../configs/run.ci.yaml) (`RUN_PROFILE` / `CI=true`) with sample-compatible tickers; local runs use [`configs/run.yaml`](../configs/run.yaml).
- **use_rolling_window** / **rolling_days**: end date = today, start = today − `rolling_days` (default in `configs/run.yaml`: **1095** ≈ 3 years; unless `MARKET_PIN_DATES=1`).

## `portfolio`

- **weighting**:
  - `equal` — equal gross weight across tickers (respecting `position_sides`).
  - `manual` — use `manual_weights` (normalized to gross budget 1).
  - `partial` — fix `anchor_weights`, optimize free tickers at Phase 3.
  - `optimised` — max-Sharpe on gross budget with optional shorts.
- **allow_shorts** / **max_gross_per_ticker**: gross-budget constraints (notebook-style).
- **position_sides**: force sign per ticker, e.g. `QQQ: short` (overrides FinBERT inference).
- **sentiment_position_sides** (default `true`): infer long/short for tickers **not** listed in `position_sides` (see [FinBERT position sides](#finbert-position-sides) below).
- **sentiment_sides_window_days** (default `30`): lookback for FinBERT side inference (matches the Signals FinBERT tab).
- **sentiment_mu_blend** (default `0.3`): blend weight α on FinBERT-based expected returns vs historical μ (see [Sentiment in optimization](#sentiment-in-optimization)).
- **sentiment_mu_mode** (`vol_scaled` \| `fixed`), **sentiment_mu_scale**, **sentiment_mu_window_days**, **sentiment_mu_hist_window_days**: μ blend knobs.
- **sentiment_magnitude_tilt** (default `true`), **sentiment_tilt_beta**, **sentiment_tilt_cap**: post–max-Sharpe gross tilt toward bullish names.
- **regime_sentiment_mix**: per HMM label (`low` / `mid` / `high`) multiplier on α and tilt strength.
- **use_legacy_bullish_mu** (default `false`): if α=0, use old full-sample `bullish_ratio` nudge instead of FinBERT blend.
- **anchor_weights**: used when `weighting: partial`; sum(|anchors|) must be &lt; 1.
- **risk_free**: annual rate for optimization Sharpe.
- **min_gross_divisor** (default `5`): minimum gross per ticker is `1 / (min_gross_divisor × n_tickers)` (long or short); applies to all weighting modes.

`aqre prepare` writes `data/raw/portfolio/holdings.parquet` and `data/run_manifest.json` (including effective `position_sides` when sentiment inference runs).

### FinBERT position sides

When **`sentiment_position_sides: true`** (default) and **`allow_shorts: true`**, the pipeline fills in `long` / `short` for any ticker **without** an entry in `position_sides`, using the **same 30-day FinBERT score** as the [Signals dashboard](signals_dashboard.md):

\[
\text{sentiment\_score} = \mathrm{mean}(\text{positive labels}) - \mathrm{mean}(\text{negative labels})
\]

| Score (30d) | Inferred side | Same regime as trailing stops |
|-------------|---------------|-------------------------------|
| ≥ **+0.3** | `long` | Bullish-derived |
| ≤ **−0.3** | `short` | Bearish-derived |
| between | *(none)* | Neutral — defaults to **long** (same as omitting the ticker) |

**Examples:** a ticker at **−0.9** (e.g. strongly bearish news) is assigned `short`; a ticker at **−0.1** stays **long** because it is above the bear threshold.

**Explicit config always wins:**

```yaml
portfolio:
  allow_shorts: true
  sentiment_position_sides: true
  position_sides:
    PDD: long   # force long even if FinBERT is bearish
```

**What this does *not* do:** it does not replace max-Sharpe optimization. Inferred sides only fix **sign** (long vs short). **Magnitudes** come from blended expected returns and optional [magnitude tilt](#sentiment-in-optimization) (see below).

**Where it runs:**

| Stage | Behavior |
|-------|----------|
| `aqre prepare` | Merges sides into weights for `equal` / `manual`; records `position_sides` in `data/run_manifest.json` |
| `optimize_portfolios` | Merges sides again, then max-Sharpe / partial; signs applied so bearish names can appear with **negative** weights in `optimal_weights.parquet` |

**Prerequisites:** `data/processed/sentiment.parquet` (Phase 1: ingest → preprocess → sentiment). If the file is missing or a ticker has no news in the window, that ticker keeps the default **long** side.

**Disable inference:** `sentiment_position_sides: false` (or `allow_shorts: false`).

**Code:** [`src/portfolio/sentiment_sides.py`](../src/portfolio/sentiment_sides.py) — `effective_position_sides()`, thresholds from [`src/portfolio/stops.py`](../src/portfolio/stops.py).

### Sentiment in optimization

For `weighting: optimised` or `partial`, `optimize_portfolios` builds expected returns in [`src/portfolio/expected_returns.py`](../src/portfolio/expected_returns.py):

\[
\mu_i = (1 - \alpha_{\mathrm{eff}})\,\mu^{\mathrm{hist}}_i + \alpha_{\mathrm{eff}}\,\mu^{\mathrm{sent}}_i,\quad
\alpha_{\mathrm{eff}} = \texttt{sentiment\_mu\_blend} \times m(\text{HMM regime})
\]

- \(\mu^{\mathrm{hist}}\): mean daily returns from `risk_dataset.parquet` (optional cap via `sentiment_mu_hist_window_days`).
- \(\mu^{\mathrm{sent}}\): 30d FinBERT `sentiment_score` mapped to daily units — `vol_scaled` → `sentiment_mu_scale × score × σ_i`, or `fixed` → `sentiment_mu_scale × score × 10⁻⁴`.
- \(m(\text{regime})\): from **`regime_sentiment_mix`** and latest label in `data/risk/portfolio/regimes.parquet` ([`src/portfolio/regime_policy.py`](../src/portfolio/regime_policy.py)). Example default: `low: 0.4`, `mid: 0.75`, `high: 1.0`.

**Post-optimization tilt** ([`src/portfolio/sentiment_tilt.py`](../src/portfolio/sentiment_tilt.py)): when `sentiment_magnitude_tilt: true`, max-Sharpe weights are scaled by z-scored FinBERT scores (`sentiment_tilt_beta`, capped by `sentiment_tilt_cap`), then re-normalized to gross budget 1 with min gross floors.

**Stacking with position sides:** sides (±0.3) → blended μ → max-Sharpe → magnitude tilt. Tune α and β moderately to avoid double-counting sentiment.

**Metadata:** `data/risk/optimization/_metadata.json` records `regime`, `alpha_eff`, `beta_eff`, `mu_mode`, `tilt_applied`.

**CI profile:** [`configs/run.ci.yaml`](../configs/run.ci.yaml) sets `sentiment_mu_blend: 0` and `sentiment_magnitude_tilt: false` for stable sample runs.

**Disable:** `sentiment_mu_blend: 0`, `sentiment_magnitude_tilt: false`, or `use_legacy_bullish_mu: true` for the old `bullish_ratio` nudge only.

### Weights vs dashboard

| Mode | `holdings.parquet` after `prepare` | Weights shown in Streamlit / copilot |
|------|-----------------------------------|--------------------------------------|
| `equal`, `manual` | Final weights | Same file |
| `optimised`, `partial` | Equal **placeholder** in holdings (for early DVC only) | `optimal_weights.parquet` after `optimize_portfolios`; VaR/regimes/backtest run **after** optimize in DVC order |

See [portfolio_dashboard.md](portfolio_dashboard.md) for the Portfolio page and `load_portfolio_weights()`.

**Stale holdings:** if `holdings.parquet` exists but tickers no longer match the active profile (e.g. after merging `main` or switching `run.yaml` ↔ `run.ci.yaml`), risk stages call `ensure_holdings()` and rewrite equal weights for the current universe. Custom weights from a prior `aqre prepare` are kept when file tickers still match.

## `research`

- **backtest_weight_source**: portfolio label in optimization output (default `max_sharpe`).

## Files produced

| Path | Purpose |
|------|---------|
| `data/raw/portfolio/holdings.parquet` | Weights from `prepare` (`equal` / `manual`); placeholder equal weights for `optimised` / `partial` until optimization |
| `data/run_manifest.json` | Run metadata (gitignored) |
| `data/risk/optimization/optimal_weights.parquet` | Weights used by dashboard, backtests, and analytics when `weighting` is `optimised` or `partial` |
