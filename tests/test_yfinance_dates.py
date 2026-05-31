"""yfinance date clamping helpers."""

from datetime import date

from src.market.adapters.yfinance import clamp_download_dates


def test_clamp_download_dates_caps_future_end() -> None:
    start, end = clamp_download_dates("2024-01-01", "2099-12-31", today=date(2024, 6, 15))  # type: ignore[call-arg]
    assert start == "2024-01-01"
    assert end == "2024-06-16"
