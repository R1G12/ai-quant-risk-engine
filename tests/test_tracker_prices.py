"""Tests for tracker price loading."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.portfolio.tracker.prices import MarkPriceInfo, _yfinance_last_close, close_panel, latest_mark_prices_with_info
from src.utils.config import load_app_config


def test_close_panel_from_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    market_dir = tmp_path / "data" / "processed" / "market" / "year=2025" / "month=06"
    market_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 6, 1, tzinfo=timezone.utc),
                datetime(2025, 6, 2, tzinfo=timezone.utc),
            ],
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 101.0],
        }
    ).write_parquet(market_dir / "part.parquet")

    glob = str(tmp_path / "data" / "processed" / "market" / "**" / "*.parquet")
    monkeypatch.setattr("src.portfolio.tracker.prices.market_processed_glob", lambda: glob)
    monkeypatch.setattr(
        "src.portfolio.tracker.prices._processed_market_available",
        lambda: True,
    )

    panel = close_panel([ "AAPL"], date(2025, 6, 1), date(2025, 6, 2))
    assert panel.height == 2
    assert float(panel.filter(pl.col("date") == date(2025, 6, 2))["close"][0]) == 101.0


def test_yfinance_last_close_uses_yesterday_end() -> None:
    yesterday = date.today() - timedelta(days=1)
    import pandas as pd

    fake = pd.DataFrame(
        {"Close": [50.0, 51.0]},
        index=pd.to_datetime([yesterday - timedelta(days=1), yesterday]),
    )
    with patch("src.portfolio.tracker.prices.download_symbol_history", return_value=fake):
        price, as_of = _yfinance_last_close("AAPL", lookback_days=7)
    assert price == pytest.approx(51.0)
    assert as_of == yesterday


def test_close_panel_yfinance_fallback_for_missing_ticker() -> None:
    import pandas as pd

    fake = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2025-06-01", "2025-06-02"]),
    )
    with patch("src.portfolio.tracker.prices._processed_market_available", return_value=False):
        with patch("src.portfolio.tracker.prices.download_symbol_history", return_value=fake):
            panel = close_panel(["ZZZ"], date(2025, 6, 1), date(2025, 6, 2))
    assert panel.height == 2
    assert panel["ticker"].unique().to_list() == ["ZZZ"]


def test_latest_mark_prices_processed_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mark_day = date.today() - timedelta(days=2)
    market_dir = tmp_path / "data" / "processed" / "market" / f"year={mark_day.year}" / f"month={mark_day.month:02d}"
    market_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(mark_day.year, mark_day.month, mark_day.day, tzinfo=timezone.utc)],
            "ticker": ["ZZZ"],
            "close": [42.0],
        }
    ).write_parquet(market_dir / "part.parquet")

    glob = str(tmp_path / "data" / "processed" / "market" / "**" / "*.parquet")
    monkeypatch.setattr("src.portfolio.tracker.prices.market_processed_glob", lambda: glob)
    monkeypatch.setattr(
        "src.portfolio.tracker.prices._processed_market_available",
        lambda: True,
    )

    app = load_app_config()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)
    with patch("src.portfolio.tracker.prices._yfinance_last_close") as yf_mock:
        info = latest_mark_prices_with_info(["ZZZ"], app)
        if close_panel(["ZZZ"], start, end).height:
            yf_mock.assert_not_called()

    assert info["ZZZ"].source == "processed"
    assert info["ZZZ"].price == pytest.approx(42.0)
