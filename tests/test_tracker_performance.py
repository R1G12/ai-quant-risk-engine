"""Tests for Tracker performance curves."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import pytest

from src.portfolio.tracker.performance import (
    actual_equity_curve,
    benchmark_equity_curve,
    tracker_comparison_bounds,
)
from src.portfolio.tracker.schema import validate_and_normalize
from src.utils.config import load_app_config


def _trades(rows: list[dict]) -> pl.DataFrame:
    return validate_and_normalize(pl.DataFrame(rows))


def test_actual_equity_rises_with_price() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": "2025-01-01",
                "side": "long",
                "action": "buy",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    panel = pl.DataFrame(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 2)],
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 110.0],
        }
    )
    curve, first = actual_equity_curve(
        trades, panel, initial_nav=1000.0, start=date(2025, 1, 1), end=date(2025, 1, 2)
    )
    assert first == date(2025, 1, 1)
    assert curve.height == 2
    assert float(curve["equity"][-1]) > float(curve["equity"][0])


def test_actual_equity_short_open_flat_nav() -> None:
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "SPY",
                "trade_date": "2025-01-01",
                "side": "short",
                "action": "sell",
                "quantity": 10,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "S1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    panel = pl.DataFrame(
        {
            "date": [date(2025, 1, 1), date(2025, 1, 2)],
            "ticker": ["SPY", "SPY"],
            "close": [100.0, 100.0],
        }
    )
    curve, first = actual_equity_curve(
        trades, panel, initial_nav=1000.0, start=date(2025, 1, 1), end=date(2025, 1, 2)
    )
    assert first == date(2025, 1, 1)
    assert float(curve["equity"][0]) == pytest.approx(1000.0)
    assert float(curve["equity"][1]) == pytest.approx(1000.0)


def test_actual_equity_empty_trades() -> None:
    trades = pl.DataFrame(
        schema={
            "trade_id": pl.Utf8,
            "ticker": pl.Utf8,
            "trade_date": pl.Date,
            "side": pl.Utf8,
            "action": pl.Utf8,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "fees": pl.Float64,
            "lot_id": pl.Utf8,
            "rolled_from_lot_id": pl.Utf8,
            "notes": pl.Utf8,
        }
    )
    panel = pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "close": pl.Float64})
    curve, first = actual_equity_curve(
        trades, panel, initial_nav=1000.0, start=date(2025, 1, 1), end=date(2025, 1, 2)
    )
    assert curve.is_empty()
    assert first is None


def test_benchmark_equity_curve_indexed() -> None:
    app = load_app_config()
    wide = pl.DataFrame(
        {
            "timestamp": [
                date(2025, 1, 1),
                date(2025, 1, 2),
                date(2025, 1, 3),
            ],
            "SPY": [0.01, -0.005, 0.02],
        }
    )
    with patch("src.portfolio.tracker.performance.load_returns_wide", return_value=wide):
        curve = benchmark_equity_curve(app, date(2025, 1, 1), date(2025, 1, 3))
    assert curve.height == 3
    assert float(curve["equity_indexed"][0]) == pytest.approx(100.0)
    assert float(curve["equity_indexed"][-1]) != 100.0


def test_benchmark_equity_curve_yfinance_fallback() -> None:
    app = load_app_config()
    yf_rets = pl.DataFrame(
        {
            "date": [date(2025, 1, 2), date(2025, 1, 3)],
            "daily_return": [0.01, 0.02],
        }
    )
    with patch("src.portfolio.tracker.performance.load_returns_wide", return_value=pl.DataFrame()):
        with patch(
            "src.portfolio.tracker.performance._benchmark_returns_from_yfinance",
            return_value=yf_rets,
        ):
            curve = benchmark_equity_curve(app, date(2025, 1, 2), date(2025, 1, 3))
    assert curve.height == 2


def test_tracker_comparison_bounds_extends_to_yesterday() -> None:
    app = load_app_config()
    yesterday = date.today() - timedelta(days=1)
    trades = _trades(
        [
            {
                "trade_id": "1",
                "ticker": "AAPL",
                "trade_date": yesterday.isoformat(),
                "side": "long",
                "action": "buy",
                "quantity": 1,
                "price": 100.0,
                "fees": 0.0,
                "lot_id": "L1",
                "rolled_from_lot_id": None,
                "notes": None,
            },
        ]
    )
    with patch("src.portfolio.tracker.performance.detect_data_bounds") as mock_bounds:
        mock_bounds.return_value = __import__(
            "src.analytics.data_bounds", fromlist=["DataBounds"]
        ).DataBounds(min_date=date(2025, 1, 1), max_date=date(2025, 6, 1), source_label="test")
        bounds = tracker_comparison_bounds(trades, app)
    assert bounds.max_date >= yesterday
    assert bounds.min_date <= yesterday
