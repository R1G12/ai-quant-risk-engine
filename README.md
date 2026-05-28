# AI Quant Risk Engine

Institutional-style **financial sentiment + risk modeling** platform for a hedge-fund class project.

- **Phase 1:** FinBERT sentiment pipeline (news → preprocess → sentiment)
- **Phase 2:** Market data lake + lazy Polars feature engineering → `risk_dataset.parquet`
- **Phase 3:** Quantitative risk engine — VaR/CVaR, vol, correlations, Markowitz optimization, efficient frontier
- **Phase 4:** Research engine — Monte Carlo paths, stress/scenarios, backtesting, experiment tracking

## Pipeline map: 7 conceptual steps vs 24 DVC stages

```mermaid
flowchart LR
  subgraph done [Implemented]
    S1[1_download_data]
    S2[2_preprocess]
    S3[3_sentiment]
    S4[4_features]
    S5[5_risk_modeling]
    S6[6_portfolio_opt]
  end
  subgraph done2 [Phase4]
    S7[7_research_eval]
  end
  S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
```

| Conceptual step | Status | DVC stage(s) |
|-----------------|--------|--------------|
| 1–4 | Done | Phase 1 + Phase 2 (10 stages) |
| 5. Risk modeling | Done | `generate_volatility_metrics`, `generate_var_metrics`, `generate_cvar_metrics`, `generate_correlations` |
| 6. Portfolio optimization | Done | `generate_portfolio_metrics`, `optimize_portfolios`, `generate_efficient_frontier` |
| 7. Research & evaluation | Done | `generate_simulations` … `generate_research_reports` |

### DVC stage flow

```mermaid
flowchart TB
  ingest[ingest] --> preprocess[preprocess] --> sentiment[sentiment]
  ingest_m[ingest_market_data] --> clean_m[clean_market_data]
  clean_m --> gen_ret[generate_returns]
  gen_ret --> gen_vol[generate_volatility_features]
  gen_ret --> gen_tech[generate_technical_features]
  sentiment --> gen_sent[generate_sentiment_features]
  gen_vol --> merge[merge_features]
  gen_tech --> merge
  gen_sent --> merge
  merge --> risk_vol[generate_volatility_metrics]
  risk_vol --> risk_var[generate_var_metrics]
  risk_var --> risk_cvar[generate_cvar_metrics]
  merge --> risk_corr[generate_correlations]
  risk_corr --> risk_port[generate_portfolio_metrics]
  risk_port --> risk_opt[optimize_portfolios]
  risk_opt --> risk_front[generate_efficient_frontier]
  risk_front --> gen_sim[generate_simulations]
  gen_sim --> run_bt[run_backtests]
  gen_sim --> run_stress[run_stress_tests]
  run_stress --> run_scen[run_scenario_analysis]
  run_bt --> eval_perf[evaluate_performance]
  gen_sim --> eval_perf
  eval_perf --> cmp_exp[compare_experiments]
  eval_perf --> gen_rep[generate_research_reports]
```

## Features

### Phase 1
- DVC: `ingest` → `preprocess` → `sentiment`
- FinBERT via `transformers.pipeline`

### Phase 2
- Market ingestion (`sample` | `yfinance`)
- Hive-partitioned OHLCV: `data/processed/market/year=YYYY/month=M/`
- Lazy writes via `sink_parquet` (Polars >= 1.20, `PartitionBy`)
- Merged output: `data/features/merged/risk_dataset.parquet`

### Phase 3
- Volatility: rolling, EWMA, regime flags, GARCH
- VaR / CVaR: historical, parametric, Monte Carlo
- Correlation/covariance with Ledoit-Wolf shrinkage
- HMM regimes, portfolio analytics
- Markowitz min-var / max-Sharpe + efficient frontier
- Plotly dashboard: `data/analytics/risk/`

## Project structure

```
ai-quant-risk-engine/
├── configs/              # YAML + schemas/
├── data/
│   ├── raw/market/
│   ├── processed/market/
│   ├── features/
│   ├── risk/
│   └── external/
├── docs/
├── src/
│   ├── ingestion/
│   ├── market/
│   ├── features/
│   ├── sentiment/
│   └── utils/
├── dvc.yaml
└── params.yaml
```

## Setup

**Python 3.12 only** (CI and local dev). Do not use Python 3.14 — many wheels (NumPy, torch) are not ready yet.

Non-technical users: see [docs/quickstart_nontechnical.md](docs/quickstart_nontechnical.md).

```powershell
cd ai-quant-risk-engine
py -3.12 -m venv .venv312
.venv312\Scripts\activate
pip install -e ".[dev,market,risk]"
copy .env.example .env
```

Requires **Polars >= 1.20** for streaming partitioned parquet writes.

Cursor/VS Code picks `.venv312` via [`.vscode/settings.json`](.vscode/settings.json). `pytest` fails fast on the wrong interpreter ([`tests/conftest.py`](tests/conftest.py)).

## Interactive dashboard (Streamlit)

Primary UX for exploring charts with a **date slider**, KPI cards, and optional **live yfinance** prices.

```powershell
.venv312\Scripts\activate
pip install -e ".[dashboard,market,research,risk]"
streamlit run src/analytics/streamlit_dashboard.py
```

| Action | Typical time |
|--------|----------------|
| Open dashboard (sample parquet) | &lt; 1–3 s |
| Move date slider / change chart | 1–3 s |
| Background yfinance (5 tickers, ~1Y) | ~15–45 s |
| Full `dvc repro` + sentiment | 10–30+ min |

- **Sample (default):** reads local DVC artifacts under `data/`.
- **Live:** background fetch to `data/cache/market_live/` (gitignored); button enables when ready. Risk/backtest metrics still come from the last sample pipeline run.

If live fetch fails with *"yfinance returned no data"*, upgrade and retry:

```powershell
pip install -U "yfinance>=1.3.0" "curl_cffi>=0.15"
```

Yahoo often blocks plain HTTP clients; `curl_cffi` mimics a browser. Also check DNS/network access to `fc.yahoo.com`, then use **Retry live fetch** in the sidebar.

Live fetch tries several Yahoo client strategies automatically (including a no-verify SSL fallback for common Windows curl cert issues). For stricter SSL, set `$env:YFINANCE_SSL_VERIFY = "1"` only. To prefer no-verify first (dev): `$env:YFINANCE_SSL_VERIFY = "0"`.

Static export (reports/CI): `data/analytics/dashboard/index.html` after `generate_research_reports`.

### Refresh sample data to ~1 year

After enabling rolling dates in config, run market + downstream stages once:

```powershell
$env:MARKET_SOURCE = "sample"
# Local dev uses rolling window (end=today, start=today-365d) unless MARKET_PIN_DATES=1
dvc repro ingest_market_data clean_market_data generate_returns merge_features
# Add risk + research stages as needed (FinBERT sentiment is the slow step)
```

Until repro completes, the slider clamps to whatever exists on disk (e.g. 2024 Q1) with a banner in the app.

## Configure tickers and date range

Edit [`params.yaml`](params.yaml) or [`configs/market.yaml`](configs/market.yaml):

```yaml
market:
  source: sample          # or yfinance
  tickers: [AAPL, MSFT, XOM, GS, JPM]
  use_rolling_window: true   # end=today, start=today-rolling_days
  rolling_days: 365
  # Pin for CI / reproducibility:
  # use_rolling_window: false
  # start_date: "2024-01-01"
  # end_date: "2024-03-31"
```

Or per session:

```powershell
$env:MARKET_SOURCE = "yfinance"
dvc repro ingest_market_data clean_market_data
```

Sentiment source → ticker mapping: [`configs/sentiment_map.yaml`](configs/sentiment_map.yaml).

## Phase 5 — Portfolio intelligence platform

Institutional dashboard, copilot, API, monitoring, and reports on top of Phases 1–4.

```powershell
pip install -e ".[dashboard,platform,risk,research]"
aqre dashboard                    # multi-page platform UI
aqre dashboard --legacy           # Phase 4 chart explorer
aqre copilot ask "Which assets contribute most to VaR?"
aqre api serve                    # FastAPI on :8000
aqre workflow run platform_reports
dvc repro generate_platform_reports
```

| Component | Location |
|-----------|----------|
| Platform dashboard | `src/dashboards/app.py` + `pages/` |
| Copilot | `src/copilot/` |
| REST API | `src/api/` |
| Monitoring | `src/monitoring/` |
| Reports | `reports/` (generated) |
| Docs | `docs/system_architecture.md`, `docs/copilot_architecture.md`, … |

## CLI (`aqre`)

After `pip install -e ".[dev]"`, use the unified CLI for common workflows:

```powershell
aqre config show
aqre run phase2 --market-source sample
aqre run phase3 --market-source sample --dry-run
aqre run stage merge_features
aqre run all
aqre dashboard
```

Environment overrides work as before (`MARKET_SOURCE`, `MARKET_PIN_DATES`, `NEWS_SOURCE`).

## Run pipelines

```bash
dvc repro
```

Phase 2 only (sample, no network):

```powershell
$env:MARKET_SOURCE = "sample"
dvc repro ingest_market_data clean_market_data generate_returns
dvc repro generate_volatility_features generate_technical_features generate_sentiment_features merge_features
```

Phase 3 risk engine (after `merge_features`):

```powershell
.venv312\Scripts\activate
$env:MARKET_SOURCE = "sample"
dvc repro generate_volatility_metrics generate_var_metrics generate_cvar_metrics
dvc repro generate_correlations generate_portfolio_metrics optimize_portfolios generate_efficient_frontier
```

## DVC remote storage (optional)

**You do not need `dvc push` for local work.** `dvc repro` already stores artifacts in `.dvc/cache` on your machine.

| Approach | When to use |
|----------|-------------|
| **No remote** (default) | Solo / class project; re-run `dvc repro` to regenerate data |
| **Local folder remote** | Practice “real” MLOps without cloud — good simulation on one PC |
| **Cloud remote** (S3, GCS, Azure, SSH) | Team sharing of large artifacts |

There is **no DVC account** to create. DVC is open-source; remotes are just storage locations.

### Local remote (configured)

Artifacts are pushed to a folder **outside the repo** (simulates cloud storage):

`D:\Romain\Projects\Finance\02_Risk Analysis\DVC_test_quant`

Configured in [`.dvc/config`](.dvc/config) as remote `localstore` (default). After `dvc repro`:

```powershell
.venv312\Scripts\activate
dvc push    # upload cache → DVC_test_quant
dvc pull    # restore on another machine / fresh clone
```

### Cloud remote (later)

```powershell
dvc remote add -d myremote s3://my-bucket/path
# or: gs://..., azure://..., ssh://user@host/path
dvc push
```

See [docs/mlops.md](docs/mlops.md) for more detail.

## Testing

```bash
pytest
# Phase 5 platform smoke (copilot, API, reports, monitoring):
pytest tests/test_phase5_integration.py tests/test_copilot_explanations.py -v
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data architecture](docs/data_architecture.md)
- [Polars guidelines](docs/polars_guidelines.md)
- [Feature store](docs/feature_store.md)
- [MLOps](docs/mlops.md)
- [Agents guide](docs/agents.md)
- [Roadmap](docs/roadmap.md)
- [Risk models](docs/risk_models.md)
- [Portfolio theory](docs/portfolio_theory.md)
- [Quantitative methods](docs/quantitative_methods.md)
- [Optimization pipeline](docs/optimization_pipeline.md)

## Roadmap

- **Phase 2b:** VaR, GARCH, portfolio optimization in `src/risk` / `src/portfolio`
- **Phase 3:** Private-markets Monte Carlo / quantum research
