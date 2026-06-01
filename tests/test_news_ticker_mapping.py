"""Tests for ticker-first news mapping."""

from __future__ import annotations

import polars as pl

from src.ingestion.news_schema import apply_ticker_mapping
from src.utils.config import load_app_config


def test_apply_ticker_mapping_prefers_ticker_column() -> None:
    app = load_app_config()
    lf = pl.LazyFrame(
        {
            "date": ["2026-01-01"],
            "ticker": ["nvda"],
            "source": ["Bloomberg"],
            "text": ["headline"],
        }
    )
    out = apply_ticker_mapping(lf, app).collect()
    assert out["ticker"].to_list() == ["NVDA"]


def test_apply_ticker_mapping_falls_back_to_sentiment_map() -> None:
    app = load_app_config()
    lf = pl.LazyFrame(
        {
            "date": ["2026-01-01"],
            "source": ["Reuters"],
            "text": ["headline"],
        }
    )
    out = apply_ticker_mapping(lf, app).collect()
    assert out["ticker"].to_list() == ["XOM"]
