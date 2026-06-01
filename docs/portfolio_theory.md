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
| `optimised` | Max-Sharpe on gross budget (optional shorts); signs from `position_sides` + FinBERT when `sentiment_position_sides` |

**Artifacts:**

- `data/raw/portfolio/holdings.parquet` — written by `aqre prepare`; final weights for `equal` / `manual`; placeholder for `optimised` / `partial` before optimization.
- `data/risk/optimization/optimal_weights.parquet` — weights for `optimised` / `partial` (see `research.backtest_weight_source`, default `max_sharpe`).

**Dashboard:** the Phase 5 [Portfolio](portfolio_dashboard.md) page and copilot call `load_portfolio_weights()` so the UI matches `portfolio.weighting`, not always equal weights from holdings.

**Trailing stops (dashboard):** sentiment-derived 3-tranche levels per ticker are on the [Signals](signals_dashboard.md) page; policy lives in `src/portfolio/stops.py` (aligned with the trading notebook).

Optimization methods in `src/risk/optimization/`:

- **Minimum variance** — `scipy.optimize.minimize` on w'Σw
- **Maximum Sharpe** — maximize (μ'w - r_f) / sqrt(w'Σw)
- **Efficient frontier** — target-return equality constraints

## Constraints

Merged from run profile into `configs/risk/optimization.yaml` at load time: `allow_shorts`, `max_gross_per_ticker`, long-only flags, and legacy max weight caps.

**Minimum gross per name:** `|w_i| ≥ 1 / (min_gross_divisor × n)` (default divisor `5`), enforced in `src/portfolio/weights.py` for `equal`, `manual`, `optimised`, and `partial`.

**Long/short sign (FinBERT):** when `sentiment_position_sides` is enabled (default), tickers without an explicit `position_sides` entry get `short` if their 30d FinBERT score ≤ −0.3 and `long` if ≥ +0.3 (same bands as trailing stops in `src/portfolio/stops.py`). Signs are merged in `aqre prepare` and `optimize_portfolios` via `effective_position_sides()` in `src/portfolio/sentiment_sides.py`; max-Sharpe then respects them with `apply_position_sides`. See [run_profile.md — FinBERT position sides](run_profile.md#finbert-position-sides).
