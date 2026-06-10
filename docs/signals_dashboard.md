# Signals dashboard (Phase 5)

The **Signals** page is a sidebar module in the Phase 5 Streamlit platform. It surfaces trading-style controls aligned with [`notebooks/trading_risk_manager_final.ipynb`](../notebooks/trading_risk_manager_final.ipynb): FinBERT sentiment per ticker, three-tranche trailing stops (vol-scaled by default), the current HMM regime, and portfolio VaR 95%.

## How to open it

```powershell
aqre run profile --dashboard
# or after a prior pipeline run:
aqre dashboard
```

In the left sidebar, select **Signals** (`src/dashboards/pages/7_Signals.py`).

The legacy chart explorer (`aqre dashboard --legacy` → `src/analytics/streamlit_dashboard.py`) does **not** include this page.

## Prerequisites (DVC artifacts)

| Tab / metric | Required pipeline output | DVC stages (minimum) |
|--------------|--------------------------|----------------------|
| FinBERT scores | `data/processed/sentiment.parquet` | `ingest` → `preprocess` → `sentiment` |
| Trailing stops (sentiment) | Same + holdings tickers in `configs/run.yaml` | As above |
| Trailing stops (`mode: vol_scaled`) | `data/features/volatility/volatility.parquet` | `generate_volatility_features` (included in full `aqre run profile`) |
| HMM regime | `data/risk/portfolio/regimes.parquet` | `generate_portfolio_metrics` |
| VaR 95% | `data/risk/var/var_metrics.parquet` (or research performance summary) | `generate_var_metrics` |

**News ingest:** `mode: live` sets `NEWS_SOURCE=yfinance` (20–60 headlines per ticker, last 30 days, written to `data/raw/news.parquet`). `mode: demo` / CI uses `sample` news. Install market extras for Yahoo: `pip install -e ".[market]"`.

**Vol-scaled stops:** if `volatility.parquet` is missing or a ticker has no row, the dashboard uses `fallback_daily_vol` from `configs/run.yaml` (default `0.02`). Run a full profile at least once so per-ticker vol reflects the live universe.

Full profile run:

```powershell
aqre prepare
aqre run profile --dashboard
```

## Page layout

Three tabs:

### FinBERT

- **30-day window** of FinBERT-labelled news, aggregated per holding ticker.
- **Ticker mapping:** uses the `ticker` column when present (yfinance ingest); otherwise `source` → ticker via [`configs/sentiment_map.yaml`](../configs/sentiment_map.yaml) (sample news).
- **Sentiment score** (per ticker): `mean(positive labels) − mean(negative labels)`.
- **Visuals**: bar chart of scores; table (`bullish_ratio`, `negative_ratio`, `article_count`, `avg_confidence`).
- **Recent chart**: slider **1–7 days** (`signals_recent_days`); shared calendar window for all holdings with forward-filled scores when a day has no new articles.
- **Stale data**: tickers with no articles in the window show an error line; tickers with older last-news dates show a **table** (`ticker`, `Last news day`) instead of blocking the chart.

### Trailing stops

Derived from each ticker’s 30-day sentiment score using [`src/portfolio/stops.py`](../src/portfolio/stops.py). **Configure in [`configs/run.yaml`](../configs/run.yaml)** under `portfolio.trailing_stops`.

**Modes**

| `mode` | Behaviour |
|--------|-----------|
| `static` (default in code) | Fixed drawdown levels per sentiment bucket (notebook defaults below). |
| `vol_scaled` (recommended in `run.yaml`) | Per-ticker levels from rolling daily vol × √(horizon) × sentiment multiplier. |

**Static mode** (when `mode: static` or omitted):

| Sentiment score | Regime tag | Stop levels (drawdown from peak) |
|-----------------|------------|----------------------------------|
| ≥ 0.3 | Bullish-derived | −8%, −14%, −20% |
| ≤ −0.3 | Bearish-derived | −3%, −6%, −10% |
| else | Neutral-derived | −5%, −10%, −15% |

**Vol-scaled formula** (per tranche \(i\), when `mode: vol_scaled`):

\[
\text{level}_i = -\sigma_i \times \text{daily\_vol} \times \sqrt{\text{horizon\_days}} \times m_{\text{sentiment}}
\]

- `daily_vol` — latest rolling std of returns from `data/features/volatility/volatility.parquet` (fallback: `fallback_daily_vol`).
- `σ` — `tranche_sigmas` (default `1.5`, `2.5`, `3.5`).
- \(m_{\text{sentiment}}\) — `sentiment_vol_mult` for bull / neutral / bear (default `1.25` / `1.0` / `0.75`).
- Levels are clamped between `max_level` (tightest) and `min_level` (widest).

| Sentiment score | Regime tag | Sentiment vol mult (default) |
|-----------------|------------|------------------------------|
| ≥ `bull_threshold` | Bullish-derived (vol-scaled) | 1.25 |
| ≤ `bear_threshold` | Bearish-derived (vol-scaled) | 0.75 |
| else | Neutral-derived (vol-scaled) | 1.0 |

Exit **fractions** per tranche still come from `bull` / `neutral` / `bear` in config. Each tranche exits its fraction of the **remaining** position when breached (see `simulate_trailing_stops` in [`src/portfolio/weights.py`](../src/portfolio/weights.py)).

- **Tranche 1** = tightest stop (fires first): the **least negative** level (e.g. NVDA might show −9% while GLD shows −3% under `vol_scaled`).
- **Table columns** (vol-scaled): `daily_vol`, `vol_scale` (= daily_vol × √(horizon) × sentiment mult), plus three tranche levels and exit fractions.
- **Visuals**: caption with formula when `vol_scaled`; full table; heatmap; tranche-1 bar chart; per-ticker “ladder” plot.

### Regime & VaR

- **Current HMM regime**: latest `regime_label` from `data/risk/portfolio/regimes.parquet` (Gaussian HMM on portfolio returns; labels `low` / `mid` / `high` by mean return).
- **Portfolio VaR 95%**: same metric as the platform home KPI (`compute_window_kpis` in `src/analytics/dashboard_kpis.py`).
- **Visual**: regime history (**last 365 observations**), y-axis always shows all three HMM labels, plus a **days per regime** table under the chart.

**Link to portfolio weights:** the same 30d FinBERT score is used for (1) **long/short sign** when `sentiment_position_sides` is enabled, (2) **blended expected returns** at optimize time, (3) optional **magnitude tilt** after max-Sharpe. Latest HMM regime scales sentiment influence via `regime_sentiment_mix`. Details: [run_profile.md](run_profile.md#sentiment-in-optimization), [portfolio_dashboard.md](portfolio_dashboard.md).

## Code map

| Piece | Path |
|-------|------|
| Streamlit page | `src/dashboards/pages/7_Signals.py` |
| Data loaders | `src/dashboards/core/signals_loaders.py` |
| Per-ticker vol for stops | `load_latest_daily_vol_by_ticker()` in `signals_loaders.py` |
| FinBERT scores + sides | `src/portfolio/sentiment_sides.py` |
| Blended μ / regime / tilt | `src/portfolio/expected_returns.py`, `regime_policy.py`, `sentiment_tilt.py` |
| Stop policy | `src/portfolio/stops.py` (`build_stops`, `vol_scaled_stop_levels`) |
| Re-exports | `src/dashboards/core/loaders.py` (`load_signals_finbert`, `load_signals_regime`) |
| Tests | `tests/test_portfolio_stops.py`, `tests/test_signals_loaders.py`, `tests/test_signals_regime_chart.py` |

## Sentiment → ticker mapping

News rows in `data/processed/sentiment.parquet` use `source` (e.g. `Bloomberg`, `Reuters`). The dashboard maps them to tickers using `configs/sentiment_map.yaml`:

```yaml
source_to_ticker:
  Bloomberg: AAPL
  Reuters: XOM
default_ticker: MARKET
```

Only tickers in **current holdings** (`data/raw/portfolio/holdings.parquet` from `aqre prepare`) appear in the Signals tables. If a ticker has no mapped news in the 30-day window, it will not appear in the FinBERT tab.

## Extending stop rules

Edit `portfolio.trailing_stops` in **`configs/run.yaml`** (then restart the dashboard). Config-only changes do not require `dvc repro`, but **`vol_scaled` needs volatility features built at least once**.

Optional overrides in code:

```python
from src.portfolio.stops import build_stops, trailing_stop_set

stops, tag, daily_vol, vol_scale = build_stops(
    sentiment_score,
    policy,
    daily_vol=0.025,  # optional; uses fallback when omitted
)
```

Set `use_manual: true` to apply `manual` levels/fractions for all tickers regardless of sentiment.

## Related docs

- [Run profile](run_profile.md) — tickers, holdings, trailing-stop config
- [Feature store](feature_store.md) — `volatility.parquet` used by vol-scaled stops
- [Portfolio dashboard](portfolio_dashboard.md) — effective weights in the UI by `weighting` mode
- [Portfolio theory](portfolio_theory.md) — weighting and optimization
- [Risk models](risk_models.md) — VaR and HMM regimes
- [System architecture](system_architecture.md) — Phase 5 layers
