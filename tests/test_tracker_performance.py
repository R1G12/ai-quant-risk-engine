"""Tests for Tracker performance curves."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.portfolio.tracker.performance import actual_equity_curve
from src.portfolio.tracker.schema import validate_and_normalize


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
