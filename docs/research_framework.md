# Research Framework (Phase 4)

Phase 4 adds reproducible **simulation**, **backtesting**, and **experiment tracking** on top of Phase 3 risk outputs.

## Layout

- `data/research/simulations/` — Monte Carlo path summaries partitioned by `experiment_id` and `simulation_type`
- `data/research/backtests/` — equity curves, rolling metrics, trades
- `data/research/stress/` — stressed tail metrics per scenario
- `data/research/scenarios/` — side-by-side scenario comparison
- `data/research/evaluation/` — aggregated performance summaries
- `experiments/` — lightweight YAML manifests (params, metrics, git hash)
- `data/analytics/research/` — Plotly HTML dashboards

## Workflow

1. **Configure:** [run_profile.md](run_profile.md) → `aqre prepare` (optional but recommended for holdings/tickers)
2. **Baseline:** `aqre run profile` or `dvc repro` through `generate_research_reports`
3. **Sweeps:** `dvc exp run <stage> -S research.simulation.n_paths=2000`

See [experiment_tracking.md](experiment_tracking.md) for DVCLive and manifest details.

## Dashboard and report

After `generate_research_reports`:

- **Primary interactive UX:** `aqre dashboard` (Phase 5 platform) or `aqre dashboard --legacy` / `streamlit run src/analytics/streamlit_dashboard.py` (Phase 4 explorer). Install `.[dashboard,market,research,risk]`. Date presets, KPI cards, sample-first with background yfinance and a **Use live data** toggle for prices.
- **Static export (CI/reports):** `data/analytics/dashboard/index.html` — iframe chart picker
- **Interpreted metrics:** `data/research/reports/REPORT.md`
- Standalone Plotly files remain under `data/analytics/research/` and `data/analytics/risk/`

### Latency expectations

| Action | Typical time |
|--------|----------------|
| Open Streamlit on existing parquet | &lt; 1–3 s |
| Date slider / chart change | 1–3 s |
| Background yfinance (~5 tickers, 1Y) | ~15–45 s |
| Full pipeline repro (incl. FinBERT) | 10–30+ min |

### Refresh ~1 year of sample artifacts

Local dev defaults to a **rolling window** (`use_rolling_window: true` in `params.yaml`). To materialize new parquet:

```powershell
$env:MARKET_SOURCE = "sample"
# Or: aqre run profile --pin-dates
dvc repro ingest_market_data clean_market_data generate_returns merge_features
# Continue through risk + research stages as needed
```

**CI** (see [mlops.md](mlops.md)): `RUN_PROFILE=configs/run.ci.yaml`, `MARKET_PIN_DATES=1`, `MARKET_SOURCE=sample`, full pipeline through `generate_research_reports`.
