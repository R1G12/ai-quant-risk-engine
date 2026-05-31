# Simulation Architecture

## Processes

| Process | Model | Notes |
|---------|-------|-------|
| GBM | dS/S = μ dt + σ dW | Log-normal; constant μ, σ per calibration window |
| Multivariate | Correlated shocks via Σ (Cholesky) | Static Σ; shrinkage from Phase 3 |
| Regime GBM | Per-regime (μ, σ) from HMM labels | Small sample → few regimes |
| Stress | vol_multiplier, drift_shift on base params | Synthetic; no live macro series |

## Implementation boundary

- **NumPy** — path generation (`src/simulation/stochastic_processes/`)
- **Polars** — summaries, parquet sinks (`paths_summary`, `tail_metrics`)

Full path tensors are **not** stored by default (`save_paths: false`).

Configure paths and seeds via [`configs/simulation/monte_carlo.yaml`](../configs/simulation/monte_carlo.yaml) and [`params.yaml`](../params.yaml) → `research.simulation`. Run profile tickers flow from [run_profile.md](run_profile.md) into market ingest before calibration.

## VaR vs path engine

`src/risk/var/monte_carlo.py` remains the Phase 3 1D Gaussian VaR engine. Path simulation lives in `src/simulation/monte_carlo/engine.py`.
