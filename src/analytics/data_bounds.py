"""Detect available timestamp span in pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from src.utils.paths import (
    MARKET_LIVE_RAW_PATH,
    RESEARCH_BACKTESTS_DIR,
    RISK_DATASET_PATH,
    RISK_PORTFOLIO_METRICS_PATH,
    RISK_PORTFOLIO_RETURNS_PATH,
)


@dataclass(frozen=True)
class DataBounds:
    """Min/max dates available in local parquet."""

    min_date: date
    max_date: date
    source_label: str


def _bounds_from_parquet(path: Path, col: str = "timestamp") -> tuple[date, date] | None:
    if not path.is_file():
        return None
    df = pl.scan_parquet(path).select(pl.col(col).min().alias("mn"), pl.col(col).max().alias("mx")).collect()
    if df.height == 0 or df["mn"][0] is None:
        return None
    mn, mx = df["mn"][0], df["mx"][0]
    return mn.date() if hasattr(mn, "date") else mn, mx.date() if hasattr(mx, "date") else mx


def detect_data_bounds(experiment_id: str = "baseline") -> DataBounds:
    """Union span from sample pipeline parquet (and live cache if present)."""
    candidates: list[tuple[date, date, str]] = []
    for path, label in (
        (RISK_DATASET_PATH, "risk_dataset"),
        (RISK_PORTFOLIO_RETURNS_PATH, "portfolio_returns"),
        (RISK_PORTFOLIO_METRICS_PATH, "portfolio_metrics"),
    ):
        bounds = _bounds_from_parquet(path)
        if bounds:
            candidates.append((*bounds, label))

    bt_path = RESEARCH_BACKTESTS_DIR / f"experiment_id={experiment_id}" / "equity_curve.parquet"
    bt_bounds = _bounds_from_parquet(bt_path)
    if bt_bounds:
        candidates.append((*bt_bounds, "backtest_equity"))

    live_bounds = _bounds_from_parquet(MARKET_LIVE_RAW_PATH)
    if live_bounds:
        candidates.append((*live_bounds, "live_yfinance"))

    if not candidates:
        today = date.today()
        return DataBounds(min_date=today, max_date=today, source_label="none")

    min_d = min(c[0] for c in candidates)
    max_d = max(c[1] for c in candidates)
    labels = ", ".join(sorted({c[2] for c in candidates}))
    return DataBounds(min_date=min_d, max_date=max_d, source_label=labels)


def slider_bounds(bounds: DataBounds, *, include_today_if_live: bool = False) -> DataBounds:
    """Extend slider max to today when live cache exists so presets can reach recent dates."""
    if not include_today_if_live:
        return bounds
    today = date.today()
    if bounds.max_date >= today:
        return bounds
    return DataBounds(
        min_date=bounds.min_date,
        max_date=today,
        source_label=bounds.source_label,
    )


def default_last_year_range(bounds: DataBounds, *, days: int = 365) -> tuple[date, date]:
    """Last N days within available data."""
    end = min(bounds.max_date, date.today())
    start = max(bounds.min_date, end - timedelta(days=days))
    return start, end
