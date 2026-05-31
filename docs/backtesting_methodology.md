# Backtesting Methodology

## Design

- Historical replay on `risk_dataset` returns (not simulated paths)
- Weights from Phase 3 optimization (`data/risk/optimization/optimal_weights.parquet`) or holdings fallback (`data/raw/portfolio/holdings.parquet`)
- Weight label driven by [run profile](run_profile.md) → `research.backtest_weight_source` (e.g. `max_sharpe`)
- Transaction costs: `tc_bps` + `slippage_bps` applied on rebalance turnover proxy

## Bias controls

| Risk | Mitigation |
|------|------------|
| Lookahead | Static weights from prior-stage optimization; `assert_no_future_timestamps` in validation |
| Survivorship | Fixed ticker list in `configs/run.yaml` / `params.yaml`; invalid symbols skipped at ingest |
| Overfitting | Walk-forward windows configurable; prefer out-of-sample slices for grading |

## Sensitivity

Run cost and volatility sweeps via DVC experiments:

```bash
dvc exp run run_backtests -S research.backtest.tc_bps=15
dvc exp run generate_simulations -S research.simulation.seed=7
```
