# Portfolio Theory

## Markowitz framework

Given expected returns μ and covariance Σ, portfolio variance is w'Σw with budget Σw = 1.

## Implemented portfolios

Configured via [run_profile.md](run_profile.md) (`configs/run.yaml`) and materialized with `aqre prepare`:

| `weighting` | Behavior |
|-------------|----------|
| `equal` | Equal gross weight (respects `position_sides`) |
| `manual` | Normalized `manual_weights` |
| `partial` | Fixed `anchor_weights`; remaining budget optimized at Phase 3 |
| `optimised` | Max-Sharpe on gross budget (optional shorts) |

Holdings for early DVC stages: `data/raw/portfolio/holdings.parquet`. Optimized weights: `data/risk/optimization/`.

**Trailing stops (dashboard):** sentiment-derived 3-tranche levels per ticker are shown on the Phase 5 [Signals](signals_dashboard.md) page; policy lives in `src/portfolio/stops.py` (aligned with the trading notebook).

Optimization methods in `src/risk/optimization/`:

- **Minimum variance** — `scipy.optimize.minimize` on w'Σw
- **Maximum Sharpe** — maximize (μ'w - r_f) / sqrt(w'Σw)
- **Efficient frontier** — target-return equality constraints

## Constraints

Merged from run profile into `configs/risk/optimization.yaml` at load time: `allow_shorts`, `max_gross_per_ticker`, long-only flags, and legacy max weight caps.
