"""Tests for yfinance news adapter (mocked, no network)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.ingestion.adapters.yfinance import (
    _extract_date,
    _extract_title,
    _normalize_item,
    load_yfinance_news,
)


def test_extract_title_nested_content() -> None:
    item = {"content": {"title": "  Rally  "}}
    assert _extract_title(item) == "Rally"


def test_extract_title_top_level_fallback() -> None:
    assert _extract_title({"title": "Flat title"}) == "Flat title"


def test_normalize_item_skips_empty_title() -> None:
    assert _normalize_item({"title": ""}, "AAPL") is None


def test_normalize_item_includes_ticker() -> None:
    row = _normalize_item(
        {
            "title": "Earnings beat",
            "summary": "Strong quarter",
            "providerPublishTime": 1704067200,
            "publisher": "Reuters",
        },
        "aapl",
    )
    assert row is not None
    assert row["ticker"] == "AAPL"
    assert row["title"] == "Earnings beat"
    assert row["content"] == "Strong quarter"


def test_extract_date_from_unix() -> None:
    assert _extract_date({"providerPublishTime": 1704067200}) == "2024-01-01"


def test_load_yfinance_news_dedupes() -> None:
    duplicate = {
        "title": "Same headline",
        "summary": "Body",
        "providerPublishTime": 1704067200,
        "publisher": "Bloomberg",
    }

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.news = [duplicate, duplicate]

    with patch("yfinance.Ticker", FakeTicker):
        df = load_yfinance_news(["AAPL"], max_headlines_per_ticker=5)

    assert df.height == 1
    assert df["ticker"][0] == "AAPL"


def test_load_yfinance_news_empty_on_failure() -> None:
    class BrokenTicker:
        @property
        def news(self):
            raise RuntimeError("network down")

    with patch("yfinance.Ticker", BrokenTicker):
        df = load_yfinance_news(["AAPL"], max_headlines_per_ticker=3)

    assert df.is_empty()


def test_load_yfinance_news_no_tickers() -> None:
    df = load_yfinance_news([])
    assert df.is_empty()
