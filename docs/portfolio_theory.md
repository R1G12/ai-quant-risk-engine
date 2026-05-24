# Portfolio Theory

## Markowitz framework

Given expected returns μ and covariance Σ, portfolio variance is w'Σw with budget Σw = 1.

## Implemented portfolios

- **Equal weight** — default holdings in `data/raw/portfolio/holdings.parquet`
- **Minimum variance** — `scipy.optimize.minimize` on w'Σw
- **Maximum Sharpe** — maximize (μ'w - r_f) / sqrt(w'Σw)
- **Efficient frontier** — target-return equality constraints

## Constraints

Configured in `configs/risk/optimization.yaml`: long-only, max weight per asset, leverage cap.
