# Portfolio dashboard (Phase 5)

The **Portfolio** sidebar page shows holdings, **effective weights** from your run profile, exposures, and attribution. The home page executive summary and copilot use the same weight resolution.

## How to open it

```powershell
aqre run profile --dashboard
# or after a prior pipeline run:
aqre dashboard
```

In the left sidebar, select **Portfolio** (`src/dashboards/pages/1_Portfolio.py`).

## Which weights are shown?

Weights follow **`portfolio.weighting`** in [`configs/run.yaml`](../configs/run.yaml). The UI calls `load_portfolio_weights()` in [`src/risk/portfolio/holdings.py`](../src/risk/portfolio/holdings.py).

| `weighting` | Source in the UI | When it is written |
|-------------|------------------|-------------------|
| `equal` | `data/raw/portfolio/holdings.parquet` | `aqre prepare` (equal gross weights) |
| `manual` | `holdings.parquet` | `aqre prepare` (your `manual_weights`, normalized) |
| `partial` | `data/risk/optimization/optimal_weights.parquet` | `optimize_portfolios` (`max_sharpe` row uses partial policy) |
| `optimised` | `optimal_weights.parquet` | `optimize_portfolios` (`max_sharpe` row) |

For **`optimised`** and **`partial`**, `aqre prepare` still writes **equal placeholder** weights to `holdings.parquet` so early DVC stages can run. The dashboard (and portfolio returns / VaR / backtests) use the **optimization file** once Phase 3 has run.

### Minimum weight per name

Every weighting mode enforces a **minimum gross exposure per ticker** (long or short):

\[
|w_i| \ge \frac{1}{\texttt{min\_gross\_divisor} \times n}
\]

Default **`min_gross_divisor: 5`** in [`configs/run.yaml`](../configs/run.yaml) → for 7 names, each position is at least **~2.9%** gross (`1/35`), so the book cannot collapse into only one or two names. Tune with `portfolio.min_gross_divisor` in the run profile.

### Long / short signs (FinBERT + `position_sides`)

Negative weights on the Portfolio page require **both**:

1. **`allow_shorts: true`** in the run profile, and  
2. A **short** sign for that ticker — from explicit `position_sides` and/or **`sentiment_position_sides`** (default `true`).

With sentiment inference enabled, any ticker **not** listed in `position_sides` gets:

| 30d FinBERT score (Signals metric) | Side |
|------------------------------------|------|
| ≤ −0.3 | `short` |
| ≥ +0.3 | `long` (stored explicitly) |
| between −0.3 and +0.3 | default `long` |

So a name can look **slightly negative** on the Signals bar chart but still show a **positive** weight if its score is above −0.3 (e.g. USO ≈ −0.1). Strongly bearish names (e.g. PDD ≈ −0.9) should appear **short** after `optimize_portfolios` when `weighting: optimised`.

**Override example** — keep a bearish name long:

```yaml
portfolio:
  position_sides:
    PDD: long
```

**Turn off** automatic sides: `sentiment_position_sides: false`.

Full reference: [Run profile — FinBERT position sides](run_profile.md#finbert-position-sides). Implementation: [`src/portfolio/sentiment_sides.py`](../src/portfolio/sentiment_sides.py).

After changing sides or sentiment data, refresh optimization:

```powershell
dvc repro optimize_portfolios
# or: aqre run profile
```

### Sentiment blend and magnitude tilt

For `optimised` / `partial`, weights reflect **blended expected returns** (historical + 30d FinBERT), **HMM regime scaling** of sentiment influence, and optional **post-optimization magnitude tilt**. See [Run profile — Sentiment in optimization](run_profile.md#sentiment-in-optimization). Check `data/risk/optimization/_metadata.json` for `alpha_eff`, `regime`, and `tilt_applied`.

### Optimization row label

The parquet file stores multiple portfolios (`min_variance`, `max_sharpe`, …). The UI uses the row matching **`research.backtest_weight_source`** in `run.yaml` (default **`max_sharpe`**).

## Prerequisites

| UI section | Required output | DVC stage (minimum) |
|------------|-----------------|---------------------|
| Weights (`equal` / `manual`) | `data/raw/portfolio/holdings.parquet` | `aqre prepare` |
| Weights (`optimised` / `partial`) | `data/risk/optimization/optimal_weights.parquet` | `optimize_portfolios` (runs **before** portfolio metrics / VaR) |
| Exposures & attribution | Merged risk artifacts + portfolio returns | `generate_portfolio_metrics` after optimize |
| Sharpe / max DD (home, Signals) | Backtest `equity_curve.parquet` | `run_backtests` after optimize |
| VaR 95%, vol, drawdown charts | `var/`, `portfolio/` artifacts | `generate_*` stages after optimize (see [optimization_pipeline.md](optimization_pipeline.md)) |

Full profile:

```powershell
aqre prepare
aqre run profile --dashboard
```

If **`optimised`** or **`partial`** is set but `optimal_weights.parquet` is missing, the Portfolio page shows a **warning** and falls back to `holdings.parquet` until you run optimization.

After changing weights or sentiment settings, refresh the **full Phase 3 tail** so KPIs and charts match:

```powershell
dvc repro optimize_portfolios generate_portfolio_metrics generate_var_metrics run_backtests
# or: aqre run profile
```

## Page layout

- **Weighting mode** caption (from run profile).
- **Portfolio weights** table (asset, weight) and source line (`holdings` vs `optimization`).
- **Exposures** table (includes `weighting_mode` column).
- **Attribution** summary from copilot (`attribution_summary`).

## Code map

| Piece | Path |
|-------|------|
| Streamlit page | `src/dashboards/pages/1_Portfolio.py` |
| Effective weights | `src/risk/portfolio/holdings.py` — `load_portfolio_weights()`, `portfolio_weighting_mode()` |
| Prepare / holdings file | `src/portfolio/prepare.py` |
| FinBERT → `position_sides` | `src/portfolio/sentiment_sides.py` |
| Blended μ + regime | `src/portfolio/expected_returns.py`, `src/portfolio/regime_policy.py` |
| Magnitude tilt | `src/portfolio/sentiment_tilt.py` |
| Optimization output | `src/risk/pipeline/optimization_stage.py` |
| Copilot context | `src/copilot/context/builder.py` |

## Related docs

- [Run profile](run_profile.md) — `portfolio.weighting`, `position_sides`, `sentiment_position_sides`, `manual_weights`, `anchor_weights`
- [Portfolio theory](portfolio_theory.md) — Markowitz modes
- [Optimization pipeline](optimization_pipeline.md) — DVC order and outputs
- [Signals dashboard](signals_dashboard.md) — uses holdings **tickers** for universe (weights independent)
