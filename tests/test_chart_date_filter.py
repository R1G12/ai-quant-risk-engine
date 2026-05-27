"""DateRange filtering for chart registry."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.analytics.charts.context import DateRange, filter_by_date_range


def test_filter_by_date_range_clamps_rows() -> None:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )
    dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 4))
    out = filter_by_date_range(df, "timestamp", dr)
    assert out.height == 3
    assert out["value"].to_list() == [2, 3, 4]


def test_filter_by_date_range_utc_aware() -> None:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1, time_zone="UTC"),
                pl.datetime(2024, 1, 3, time_zone="UTC"),
                interval="1d",
                eager=True,
            ),
            "value": [10, 20, 30],
        }
    )
    dr = DateRange(start=date(2024, 1, 2), end=date(2024, 1, 3))
    out = filter_by_date_range(df, "timestamp", dr)
    assert out.height == 2
    assert out["value"].to_list() == [20, 30]
