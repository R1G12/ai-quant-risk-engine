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
| HMM regime | `data/risk/portfolio/regimes.parquet` | `optimize_portfolios` then `generate_portfolio_metrics` |
| VaR 95% | `data/risk/var/var_metrics.parquet` (or research performance summary) | `optimize_portfolios` then `generate_var_metrics` |

Full profile run:

```powershell
aqre prepare
aqre run profile --dashboard
```

## Page layout

Three tabs:

### FinBERT

- **30-day window** of FinBERT-labelled news, aggregated per holding ticker.
- With `mode: live` in [`configs/run.yaml`](../configs/run.yaml), re-run Phase 1 (`aqre run profile --phase phase1`) so ingest pulls Yahoo headlines per ticker before FinBERT.
- **Sentiment score** (per ticker):  
  `mean(positive labels) − mean(negative labels)`  
  on rows attributed by `ticker` (live yfinance) or publisher → ticker via [`configs/sentiment_map.yaml`](../configs/sentiment_map.yaml) (sample).
- **Visuals**: bar chart of scores; table (`bullish_ratio`, `negative_ratio`, `article_count`, `avg_confidence`); optional daily line chart for one ticker.

**Link to portfolio weights:** the **same 30-day score** feeds optimization in three ways (defaults in [`configs/run.yaml`](../configs/run.yaml)):

1. **Sign** (`sentiment_position_sides`): score ≤ −0.3 → short, ≥ +0.3 → long when not in `position_sides` (requires `allow_shorts: true`). Scores between ±0.3 do not force a short.
2. **Expected returns** (`sentiment_mu_blend`): blended with historical μ before max-Sharpe (`optimised` / `partial`).
3. **Magnitude tilt** (`sentiment_magnitude_tilt`): post-optimization gross tilt toward higher scores.

The **Regime & VaR** tab’s latest HMM label scales sentiment influence via `regime_sentiment_mix` (less in `low`, more in `high`). The Portfolio page shows **signed weights** after optimization, not the raw FinBERT bar. Details: [Run profile — FinBERT position sides](run_profile.md#finbert-position-sides), [Sentiment in optimization](run_profile.md#sentiment-in-optimization), [Portfolio dashboard](portfolio_dashboard.md#sentiment-blend-and-magnitude-tilt).

### Trailing stops

Derived from each ticker’s 30-day sentiment score using [`src/portfolio/stops.py`](../src/portfolio/stops.py) (same rules as the notebook).

| Sentiment score | Regime tag | Stop levels (drawdown from peak) |
|-----------------|------------|----------------------------------|
| ≥ 0.3 | Bullish-derived | -8%, -14%, -20% |
| ≤ -0.3 | Bearish-derived | -3%, -6%, -10% |
| else | Neutral-derived | -5%, -10%, -15% |

Each tranche exits **⅓** of the remaining position when breached (see `simulate_trailing_stops` in [`src/portfolio/weights.py`](../src/portfolio/weights.py) for path simulation).

- **Tranche 1** = tightest stop (fires first): e.g. **-5%** for neutral — the least negative level.
- **Visuals**: full table; heatmap of stop levels; tranche-1 bar chart; per-ticker “ladder” plot.

### Regime & VaR

- **Current HMM regime**: latest `regime_label` from `data/risk/portfolio/regimes.parquet` (Gaussian HMM on portfolio returns). Also scales `sentiment_mu_blend` and magnitude tilt via `regime_sentiment_mix` at optimization time.
- **Portfolio VaR 95%**: same metric as the platform home KPI (`compute_window_kpis` in `src/analytics/dashboard_kpis.py`).
- **Visual**: regime history (last 365 observations); y-axis always shows `low`, `mid`, and `high` even if a label has no days in the window.

## Code map

| Piece | Path |
|-------|------|
| Streamlit page | `src/dashboards/pages/7_Signals.py` |
| Data loaders | `src/dashboards/core/signals_loaders.py` |
| FinBERT window + position-side merge | `src/portfolio/sentiment_sides.py` |
| Blended μ + regime scaling | `src/portfolio/expected_returns.py`, `src/portfolio/regime_policy.py` |
| Magnitude tilt | `src/portfolio/sentiment_tilt.py` |
| Stop policy | `src/portfolio/stops.py` |
| Re-exports | `src/dashboards/core/loaders.py` (`load_signals_finbert`, `load_signals_regime`) |
| Tests | `tests/test_portfolio_stops.py`, `tests/test_signals_loaders.py` |

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

Edit constants in `src/portfolio/stops.py` or call `build_stops(score, use_manual=True, manual_levels=..., manual_fractions=...)`. Keep notebook and dashboard in sync if you change thresholds.

## Related docs

- [Run profile](run_profile.md) — tickers, holdings, [sentiment in optimization](run_profile.md#sentiment-in-optimization)
- [Portfolio dashboard](portfolio_dashboard.md) — effective weights in the UI by `weighting` mode
- [Portfolio theory](portfolio_theory.md) — weighting and optimization
- [Risk models](risk_models.md) — VaR and HMM regimes
- [System architecture](system_architecture.md) — Phase 5 layers
