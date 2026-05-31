# Risk Models (Phase 3)

## Overview

Phase 3 adds portfolio-level quantitative risk on top of `risk_dataset.parquet` (Phase 2).

## Models implemented

| Module | Outputs |
|--------|---------|
| `src/risk/volatility` | Rolling vol, EWMA vol, regime flags, GARCH vol |
| `src/risk/var` | Historical, parametric, Monte Carlo VaR |
| `src/risk/cvar` | Expected shortfall |
| `src/risk/correlations` | Covariance + rolling correlation (long format) |
| `src/risk/regimes` | HMM regime labels |
| `src/risk/optimization` | Min-variance, max-Sharpe, efficient frontier |

## Limitations

- Short samples → unstable covariance; Ledoit-Wolf shrinkage enabled by default.
- Parametric VaR assumes normality; use historical VaR for fat tails.
- GARCH/HMM require `collect()` on portfolio returns (small T).
