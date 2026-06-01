"""Tests for sentiment-derived trailing stop helpers."""

from __future__ import annotations

import polars as pl
import pytest

from src.portfolio.stops import (
    BEAR_STOP_LEVELS,
    BULL_STOP_LEVELS,
    NEUTRAL_STOP_LEVELS,
    build_stops,
    trailing_stop_set,
)
from src.dashboards.core.signals_loaders import load_trailing_stops_table


def test_build_stops_bull() -> None:
    stops, tag = build_stops(0.5)
    assert tag == "Bullish-derived"
    assert [s["level"] for s in stops] == list(BULL_STOP_LEVELS)


def test_build_stops_neutral() -> None:
    stops, tag = build_stops(0.0)
    assert tag == "Neutral-derived"
    assert [s["level"] for s in stops] == list(NEUTRAL_STOP_LEVELS)


def test_build_stops_bear() -> None:
    stops, tag = build_stops(-0.5)
    assert tag == "Bearish-derived"
    assert [s["level"] for s in stops] == list(BEAR_STOP_LEVELS)


def test_tranche_1_is_tightest_level() -> None:
    tset = trailing_stop_set(0.0)
    assert tset.tranche_1_level == pytest.approx(-0.05)
    assert tset.tranche_1_level == max(s["level"] for s in tset.stops)


def test_load_trailing_stops_table() -> None:
    summary = pl.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "sentiment_score": [0.5, -0.5],
            "bullish_ratio": [0.6, 0.2],
            "negative_ratio": [0.1, 0.5],
            "article_count": [10, 8],
            "avg_confidence": [0.9, 0.85],
        }
    )
    out = load_trailing_stops_table(summary)
    assert out.height == 2
    assert "tranche_1_level" in out.columns
    aapl = out.filter(pl.col("ticker") == "AAPL").row(0, named=True)
    assert aapl["regime_tag"] == "Bullish-derived"
