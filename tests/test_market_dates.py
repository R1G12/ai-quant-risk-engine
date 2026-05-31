"""Rolling market date resolution tests."""

from __future__ import annotations

import os
from datetime import date

from src.utils.config import resolve_market_dates


def test_resolve_market_dates_rolling() -> None:
    merged = {"use_rolling_window": True, "rolling_days": 30}
    old = os.environ.pop("MARKET_PIN_DATES", None)
    try:
        start, end = resolve_market_dates(merged, today=date(2026, 5, 23))
        assert start == "2026-04-23"
        assert end == "2026-05-23"
    finally:
        if old is not None:
            os.environ["MARKET_PIN_DATES"] = old


def test_resolve_market_dates_pinned() -> None:
    merged = {
        "use_rolling_window": False,
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
    }
    start, end = resolve_market_dates(merged, today=date(2026, 5, 23))
    assert start == "2024-01-01"
    assert end == "2024-03-31"
