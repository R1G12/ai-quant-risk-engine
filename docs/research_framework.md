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

1. **Baseline:** `dvc repro` through `generate_research_reports`
2. **Sweeps:** `dvc exp run <stage> -S research.simulation.n_paths=2000`

See `experiment_tracking.md` for DVCLive and manifest details.

## Dashboard and report

After `generate_research_reports`:

- **Interactive dashboard:** `data/analytics/dashboard/index.html` — chart picker with explanations
- **Interpreted metrics:** `data/research/reports/REPORT.md`
- Standalone Plotly files remain under `data/analytics/research/` and `data/analytics/risk/`
