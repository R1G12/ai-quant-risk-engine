"""Lazy data loaders for dashboard modules."""

from __future__ import annotations

import polars as pl

from src.analytics.charts.context import ChartContext, DateRange
from src.analytics.charts.registry import build_chart_registry
from src.analytics.dashboard_kpis import WindowKpis, compute_window_kpis
from src.analytics.data_bounds import detect_data_bounds, slider_bounds
from src.copilot.context.builder import PortfolioContext, build_portfolio_context
from src.utils.config import AppConfig, load_app_config
from src.utils.paths import display_path


def load_app() -> AppConfig:
    return load_app_config()


def load_chart_context(app: AppConfig, date_range: DateRange | None) -> ChartContext:
    return ChartContext(app=app, date_range=date_range)


def load_kpis(app: AppConfig, date_range: DateRange | None) -> WindowKpis:
    return compute_window_kpis(app, date_range)


def load_bounds(app: AppConfig):
    raw = detect_data_bounds(app.research.meta.experiment_id)
    return raw, slider_bounds(raw, include_today_if_live=False)


def load_copilot_context(app: AppConfig, date_range: DateRange | None = None) -> PortfolioContext:
    return build_portfolio_context(app, date_range=date_range)


def registry():
    return build_chart_registry()


def load_signals_finbert(app: AppConfig, *, window_days: int = 30):
    """FinBERT summary + daily scores for Signals page."""
    from src.dashboards.core.signals_loaders import load_finbert_sentiment

    return load_finbert_sentiment(app, window_days=window_days)


def load_signals_regime():
    """Latest HMM regime label and history."""
    from src.dashboards.core.signals_loaders import load_latest_regime

    return load_latest_regime()


def scan_experiments() -> pl.DataFrame:
    from src.utils.paths import EXPERIMENTS_BACKTESTS_DIR, EXPERIMENTS_SIMULATIONS_DIR, EXPERIMENTS_STRESS_DIR

    rows: list[dict[str, str]] = []
    for kind, base in (
        ("simulations", EXPERIMENTS_SIMULATIONS_DIR),
        ("backtests", EXPERIMENTS_BACKTESTS_DIR),
        ("stress_tests", EXPERIMENTS_STRESS_DIR),
    ):
        if not base.is_dir():
            continue
        for path in base.iterdir():
            if path.is_dir():
                name = path.name
                if name.startswith("experiment_id="):
                    name = name.split("=", 1)[1]
                rows.append({"kind": kind, "experiment_id": name, "path": display_path(path)})
    return pl.DataFrame(rows) if rows else pl.DataFrame({"kind": [], "experiment_id": [], "path": []})
