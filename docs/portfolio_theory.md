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

**Trailing stops (dashboard):** 3-tranche drawdown levels per ticker on the Phase 5 [Signals](signals_dashboard.md) page. Default run profile uses **`mode: vol_scaled`** (per-ticker rolling vol × sentiment multiplier); `mode: static` keeps fixed % levels per sentiment bucket. Policy: `src/portfolio/stops.py`.

Optimization methods in `src/risk/optimization/`:

- **Minimum variance** — `scipy.optimize.minimize` on w'Σw
- **Maximum Sharpe** — maximize (μ'w - r_f) / sqrt(w'Σw)
- **Efficient frontier** — target-return equality constraints

## Constraints

Merged from run profile into `configs/risk/optimization.yaml` at load time: `allow_shorts`, `max_gross_per_ticker`, long-only flags, and legacy max weight caps.

**Minimum gross per name:** `|w_i| ≥ 1 / (min_gross_divisor × n)` (default divisor `5`), enforced in `src/portfolio/weights.py` for `equal`, `manual`, `optimised`, and `partial`.

**Long/short sign (FinBERT):** when `sentiment_position_sides` is enabled (default), tickers without an explicit `position_sides` entry get `short` if their 30d FinBERT score ≤ −0.3 and `long` if ≥ +0.3. Signs are merged in `aqre prepare` and `optimize_portfolios` via `effective_position_sides()` in `src/portfolio/sentiment_sides.py`.
