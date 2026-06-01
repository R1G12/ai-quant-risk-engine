"""Ticker parsing and pipeline vs live chart classification."""

from __future__ import annotations

import re

from src.analytics.charts.base import ChartSpec
from src.utils.config import AppConfig

# Charts that can show user-selected live prices (yfinance) in Live mode.
LIVE_TICKER_CHART_IDS = frozenset({"equity_curve", "rolling_sharpe"})


def parse_ticker_input(text: str) -> list[str]:
    """Parse comma/space-separated tickers; uppercase, dedupe, drop empties."""
    if not text or not text.strip():
        return []
    parts = re.split(r"[\s,;]+", text.strip().upper())
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def pipeline_tickers(app: AppConfig) -> list[str]:
    return [t.upper() for t in app.market.tickers]


def chart_uses_pipeline_only(spec: ChartSpec, *, data_mode: str) -> bool:
    """True when sidebar tickers do not change this chart's data."""
    if data_mode == "live" and spec.id in LIVE_TICKER_CHART_IDS:
        return False
    return True


def pipeline_ticker_warning_markdown(app: AppConfig, custom_tickers: list[str]) -> str:
    """Markdown for warning when custom tickers do not apply to a chart."""
    pipe = ", ".join(pipeline_tickers(app))
    custom = ", ".join(custom_tickers) if custom_tickers else "(none)"
    return f"""
**This chart still uses the DVC sample pipeline** (not your live ticker selection).

| | Tickers |
|---|---------|
| **Pipeline (this chart)** | `{pipe}` |
| **Your selection (live prices only)** | `{custom}` |

To change backtest, risk, simulation, and stress charts to a **new universe**, update the pipeline:

1. Edit **`params.yaml`** → `market.tickers` (or `configs/market.yaml`).
2. Rebuild artifacts (PowerShell, from project root, `.venv312` active):

```powershell
$env:MARKET_PIN_DATES = $null
$env:MARKET_SOURCE = "sample"
Remove-Item -Recurse -Force data\\external\\sample_market -ErrorAction SilentlyContinue
dvc repro ingest_market_data clean_market_data generate_returns merge_features --force
dvc repro generate_correlations optimize_portfolios generate_portfolio_metrics generate_volatility_metrics generate_var_metrics generate_cvar_metrics generate_efficient_frontier run_backtests --force
dvc repro generate_simulations run_backtests run_stress_tests run_scenario_analysis evaluate_performance generate_research_reports --force
```

For **real** prices in the pipeline, set `$env:MARKET_SOURCE = "yfinance"` on the ingest step instead of `sample`.

**Live mode** + **Backtest equity** / **Rolling Sharpe** charts show **yfinance closes** for the tickers you enter in the sidebar.
"""
