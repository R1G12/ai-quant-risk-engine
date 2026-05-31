"""Tests for shared wide returns loading."""

from __future__ import annotations

import polars as pl

from src.features.wide_returns import returns_long_to_wide


def test_returns_long_to_wide_dedupes_duplicate_timestamps() -> None:
    long = pl.DataFrame(
        {
            "timestamp": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "returns": [0.01, 0.03, 0.02],
        }
    )
    wide = returns_long_to_wide(long, ["AAPL"])
    assert wide.height == 2
    assert wide["AAPL"].to_list() == [0.02, 0.02]


def test_returns_long_to_wide_multiple_tickers() -> None:
    long = pl.DataFrame(
        {
            "timestamp": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "returns": [0.01, 0.02, 0.03],
        }
    )
    wide = returns_long_to_wide(long, ["AAPL", "MSFT"])
    assert set(wide.columns) >= {"timestamp", "AAPL", "MSFT"}
    assert wide.height == 1
    assert wide["AAPL"][0] == 0.02


def test_returns_long_to_wide_empty() -> None:
    wide = returns_long_to_wide(pl.DataFrame(), [])
    assert wide.height == 0
