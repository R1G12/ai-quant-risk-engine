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
|-------------|------------------|--------------------|
| `equal` | Equal gross weights from `aqre prepare` | `holdings.parquet` |
| `manual` | Normalized `manual_weights` | `holdings.parquet` |
| `partial` / `optimised` | `data/risk/optimization/optimal_weights.parquet` (`max_sharpe`) | After `dvc repro optimize_portfolios` |

If optimization has not run, partial/optimised modes may show placeholder equal weights until Phase 3 completes.

## Minimum gross per ticker

When optimizing, each name must satisfy:

`|w_i| ≥ 1 / (min_gross_divisor × n_tickers)` (default divisor **5**).

For 7 tickers that is about **2.9%** gross each, which prevents max-Sharpe from concentrating 100% of risk in one or two names.

Configure via `portfolio.min_gross_divisor` in `configs/run.yaml`. Implemented in [`src/portfolio/weights.py`](../src/portfolio/weights.py).

## Long / short signs (FinBERT + YAML)

1. **`position_sides`** in YAML always wins for that ticker.
2. With **`sentiment_position_sides: true`** (default) and **`allow_shorts: true`**, tickers without an explicit side use 30d FinBERT score:
   - score ≤ −0.3 → `short`
   - score ≥ +0.3 → `long`
   - otherwise default `long`

Same thresholds as trailing stops in [`src/portfolio/stops.py`](../src/portfolio/stops.py). Logic: [`src/portfolio/sentiment_sides.py`](../src/portfolio/sentiment_sides.py).

## Sentiment in optimization (summary)

The same FinBERT scores also feed expected returns and optional post-opt tilt. See [run_profile.md](run_profile.md#sentiment-in-optimization) and the Signals page.

## Code map

| Piece | Path |
|-------|------|
| Streamlit page | `src/dashboards/pages/1_Portfolio.py` |
| Weight resolution | `src/risk/portfolio/holdings.py` |
| Prepare / manifest | `src/portfolio/prepare.py` |
| Optimization | `src/risk/pipeline/optimization_stage.py` |
| Min gross + optimizers | `src/portfolio/weights.py` |
| FinBERT sides + scores | `src/portfolio/sentiment_sides.py` |
| Blended μ | `src/portfolio/expected_returns.py` |

## Related docs

- [run_profile.md](run_profile.md)
- [signals_dashboard.md](signals_dashboard.md)
- [portfolio_theory.md](portfolio_theory.md)
- [optimization_pipeline.md](optimization_pipeline.md)
