# Reimplementation backlog (regression checklist)

Features restored after accidental loss during Signals/news refactors. **One logical feature per commit** when possible.

## Regression test bundle

After touching Signals loaders, portfolio optimization, or related config:

```powershell
.\.venv312\Scripts\pytest.exe tests/test_signals_regime_chart.py tests/test_min_gross_weights.py tests/test_sentiment_position_sides.py tests/test_expected_returns.py tests/test_regime_policy.py tests/test_sentiment_tilt.py tests/test_signals_loaders.py tests/test_optimization_run_profile.py -q
```

## Feature checklist

| Feature | Code | Tests | Docs | Verified |
|---------|------|-------|------|----------|
| Shared FinBERT loader | `src/portfolio/sentiment_sides.py` | `test_sentiment_position_sides.py`, `test_signals_loaders.py` | [run_profile.md](run_profile.md) | 2026-06-01 — regression bundle + full pytest (182 passed); working tree atop `46c2541` |
| Min gross `1/(5×n)` | `src/portfolio/weights.py` | `test_min_gross_weights.py` | [portfolio_theory.md](portfolio_theory.md) | 2026-06-01 — same |
| FinBERT position sides | `prepare.py`, `optimization_stage.py` | `test_sentiment_position_sides.py` | [run_profile.md](run_profile.md#finbert-position-sides) | 2026-06-01 — same |
| Blended μ + regime mixer + tilt | `expected_returns.py`, `regime_policy.py`, `sentiment_tilt.py` | `test_expected_returns.py`, `test_regime_policy.py`, `test_sentiment_tilt.py` | [run_profile.md](run_profile.md#sentiment-in-optimization) | 2026-06-01 — same |
| Regime chart (365d, fixed axis, counts) | `signals_loaders.py`, `7_Signals.py` | `test_signals_regime_chart.py` | [signals_dashboard.md](signals_dashboard.md) | 2026-06-01 — same |
| Portfolio dashboard guide | — | — | [portfolio_dashboard.md](portfolio_dashboard.md) | 2026-06-01 — same |
| Local LLM guide | — | — | [local_llm_integration.md](local_llm_integration.md) | 2026-06-01 — same |

## Agent rules

- Do **not** full-file overwrite `src/dashboards/core/signals_loaders.py` or `src/risk/pipeline/optimization_stage.py` without running the regression bundle above.
- Keep FinBERT scoring in **`sentiment_sides.py`**; dashboard loaders delegate only.
- When adding regime chart helpers, keep them in `signals_loaders.py` (or `dashboards/core/regime_chart.py`) — do not drop them in unrelated refactors.
