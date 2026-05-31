"""Chart build context and date filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.utils.config import AppConfig


@dataclass(frozen=True)
class DateRange:
    """Inclusive date window for time-series charts."""

    start: date
    end: date


@dataclass
class ChartContext:
    """Inputs for dashboard chart builders."""

    app: AppConfig
    date_range: DateRange | None = None


def filter_by_date_range(df: pl.DataFrame, col: str, dr: DateRange | None) -> pl.DataFrame:
    """Filter dataframe to [start, end] on a date/datetime column (UTC-safe)."""
    if dr is None or col not in df.columns or df.height == 0:
        return df
    # Compare calendar dates so UTC-aware and naive timestamps both work.
    return df.filter(
        pl.col(col).dt.date() >= dr.start,
        pl.col(col).dt.date() <= dr.end,
    )
