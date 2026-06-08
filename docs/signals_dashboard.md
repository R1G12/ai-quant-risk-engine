# Signals dashboard (Phase 5)

The **Signals** page is a sidebar module in the Phase 5 Streamlit platform. It surfaces trading-style controls aligned with [`notebooks/trading_risk_manager_final.ipynb`](../notebooks/trading_risk_manager_final.ipynb): FinBERT sentiment per ticker, three tranche trailing stops, the current HMM regime, and portfolio VaR 95%.

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
| Trailing stops | Same + holdings tickers in `configs/run.yaml` | As above |

**News ingest:** `mode: live` sets `NEWS_SOURCE=yfinance` (20–60 headlines per ticker, last 30 days, written to `data/raw/news.parquet`). `mode: demo` / CI uses `sample` news. Install market extras for Yahoo: `pip install -e ".[market]"`.
| HMM regime | `data/risk/portfolio/regimes.parquet` | `generate_portfolio_metrics` |
| VaR 95% | `data/risk/var/var_metrics.parquet` (or research performance summary) | `generate_var_metrics` |

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
- **Visuals**: bar chart of scores; table (`bullish_ratio`, `negative_ratio`, `article_count`, `avg_confidence`); **last 3 calendar days for all tickers** on one chart (forward-filled when a day has no new articles, with a red stale-data banner per ticker).

### Trailing stops

Derived from each ticker’s 30-day sentiment score using [`src/portfolio/stops.py`](../src/portfolio/stops.py) (same rules as the notebook). **Configure levels in [`configs/run.yaml`](../configs/run.yaml)** under `portfolio.trailing_stops`.

| Sentiment score | Regime tag | Stop levels (drawdown from peak) |
|-----------------|------------|----------------------------------|
| ≥ 0.3 | Bullish-derived | -8%, -14%, -20% |
| ≤ -0.3 | Bearish-derived | -3%, -6%, -10% |
| else | Neutral-derived | -5%, -10%, -15% |

Each tranche exits **⅓** of the remaining position when breached (see `simulate_trailing_stops` in [`src/portfolio/weights.py`](../src/portfolio/weights.py) for path simulation).

- **Tranche 1** = tightest stop (fires first): e.g. **-5%** for neutral — the least negative level.
- **Visuals**: full table; heatmap of stop levels; tranche-1 bar chart; per-ticker “ladder” plot.

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
| FinBERT scores + sides | `src/portfolio/sentiment_sides.py` |
| Blended μ / regime / tilt | `src/portfolio/expected_returns.py`, `regime_policy.py`, `sentiment_tilt.py` |
| Stop policy | `src/portfolio/stops.py` |
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

Edit `portfolio.trailing_stops` in **`configs/run.yaml`** (then restart the dashboard). Optional override in code: `build_stops(score, policy=..., use_manual=True)`.

## Related docs

- [Run profile](run_profile.md) — tickers and holdings
- [Portfolio dashboard](portfolio_dashboard.md) — effective weights in the UI by `weighting` mode
- [Portfolio theory](portfolio_theory.md) — weighting and optimization
- [Risk models](risk_models.md) — VaR and HMM regimes
- [System architecture](system_architecture.md) — Phase 5 layers
