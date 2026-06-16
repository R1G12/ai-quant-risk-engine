"""Assemble auditable portfolio context from pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from src.analytics.dashboard_kpis import WindowKpis, compute_window_kpis
from src.analytics.charts.context import DateRange
from src.risk.portfolio.exposures import exposure_table
from src.risk.portfolio.holdings import load_weights
from src.risk.portfolio.weights_loader import load_optimization_weights
from src.utils.config import AppConfig, load_app_config
from src.utils.paths import (
    RISK_CORRELATIONS_DIR,
    RISK_DATASET_PATH,
    RISK_OPT_WEIGHTS_PATH,
    RISK_PORTFOLIO_METRICS_PATH,
    RISK_REGIMES_PATH,
    RISK_VAR_DIR,
)


@dataclass
class PortfolioContext:
    """Structured snapshot for copilot reasoning (no LLM required)."""

    experiment_id: str
    tickers: list[str]
    weights: dict[str, float]
    exposures: pl.DataFrame
    kpis: WindowKpis
    var_table: pl.DataFrame | None
    correlations: pl.DataFrame | None
    regimes: pl.DataFrame | None
    portfolio_metrics: pl.DataFrame | None
    sentiment_summary: pl.DataFrame | None
    as_of: date
    metadata: dict[str, object] = field(default_factory=dict)


def _load_var_table() -> pl.DataFrame | None:
    path = RISK_VAR_DIR / "var_metrics.parquet"
    return pl.read_parquet(path) if path.is_file() else None


def _load_correlations() -> pl.DataFrame | None:
    path = RISK_CORRELATIONS_DIR / "correlations_latest.parquet"
    return pl.read_parquet(path) if path.is_file() else None


def _load_sentiment_summary(app: AppConfig, tickers: list[str]) -> pl.DataFrame | None:
    if not RISK_DATASET_PATH.is_file() or not tickers:
        return None
    return (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .group_by("ticker")
        .agg(
            pl.col("bullish_ratio").mean().alias("avg_bullish_ratio"),
            pl.col("returns").std().alias("return_vol"),
        )
        .collect()
    )


def _resolve_portfolio_weights(app: AppConfig) -> tuple[dict[str, float], str]:
    """Holdings for equal/manual; optimized weights when available."""
    weighting = (
        app.run.portfolio.weighting
        if app.run is not None
        else app.risk.portfolio.weight_mode
    )
    if weighting in ("optimised", "partial"):
        weights, label = load_optimization_weights(app)
        if not label.startswith("holdings"):
            return weights, label
    return load_weights(app), weighting


def build_portfolio_context(
    app: AppConfig | None = None,
    *,
    date_range: DateRange | None = None,
) -> PortfolioContext:
    """Build portfolio context from on-disk DVC artifacts."""
    app = app or load_app_config()
    weights, weight_source = _resolve_portfolio_weights(app)
    tickers = list(weights.keys())

    regimes = pl.read_parquet(RISK_REGIMES_PATH) if RISK_REGIMES_PATH.is_file() else None
    port_metrics = (
        pl.read_parquet(RISK_PORTFOLIO_METRICS_PATH) if RISK_PORTFOLIO_METRICS_PATH.is_file() else None
    )

    meta: dict[str, object] = {"weight_source": weight_source}
    if RISK_OPT_WEIGHTS_PATH.is_file():
        meta["optimization_weights_path"] = str(RISK_OPT_WEIGHTS_PATH)

    return PortfolioContext(
        experiment_id=app.research.meta.experiment_id,
        tickers=tickers,
        weights=weights,
        exposures=exposure_table(weights),
        kpis=compute_window_kpis(app, date_range),
        var_table=_load_var_table(),
        correlations=_load_correlations(),
        regimes=regimes,
        portfolio_metrics=port_metrics,
        sentiment_summary=_load_sentiment_summary(app, tickers),
        as_of=date.today(),
        metadata=meta,
    )
