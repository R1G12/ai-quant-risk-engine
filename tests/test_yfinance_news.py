"""Tests for yfinance news adapter (no network)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.ingestion.adapters.yfinance_news import (
    fetch_ticker_news,
    fetch_ticker_news_resilient,
    load_yfinance_news,
    yfinance_news_strategies,
)
from src.ingestion.news_schema import RAW_NEWS_COLUMNS, clamp_headline_cap


def test_clamp_headline_cap() -> None:
    assert clamp_headline_cap(10) == 20
    assert clamp_headline_cap(40) == 40
    assert clamp_headline_cap(100) == 60


def test_yfinance_news_strategies_prefers_no_verify() -> None:
    strategies = yfinance_news_strategies()
    assert strategies
    assert strategies[0][0] == "curl_cffi_no_verify"
    assert strategies[0][1] is not None


def test_normalize_nested_yfinance_14_payload() -> None:
    from src.ingestion.adapters.yfinance_news import _normalize_item

    item = {
        "id": "abc",
        "content": {
            "title": "Nested headline",
            "summary": "Nested body",
            "pubDate": "2026-06-01T13:09:15Z",
            "provider": {"displayName": "Yahoo Finance"},
        },
    }
    row = _normalize_item(item, "nvda")
    assert row is not None
    assert row["ticker"] == "NVDA"
    assert row["title"] == "Nested headline"
    assert row["source"] == "Yahoo Finance"
    assert row["date"] == "2026-06-01"


def test_fetch_ticker_news_filters_sorts_and_caps() -> None:
    today = date.today()
    old = (today - timedelta(days=40)).isoformat()
    recent = [(today - timedelta(days=i)).isoformat() for i in range(5)]

    items = [{"title": f"h{i}", "providerPublishTime": d, "publisher": "Yahoo"} for i, d in enumerate(recent)]
    items.append({"title": "old", "providerPublishTime": old, "publisher": "Yahoo"})

    mock_ticker = MagicMock()
    mock_ticker.get_news.return_value = items

    with patch("yfinance.Ticker", return_value=mock_ticker):
        rows = fetch_ticker_news("NVDA", lookback_days=30, max_headlines=3)

    assert len(rows) == 3
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates, reverse=True)
    assert all(r["ticker"] == "NVDA" for r in rows)
    mock_ticker.get_news.assert_called_once_with(count=3, tab="news")


def test_fetch_ticker_news_resilient_falls_back_on_ssl_error() -> None:
    today = date.today().isoformat()
    good_row = [
        {
            "date": today,
            "ticker": "NVDA",
            "source": "Yahoo",
            "title": "Headline",
            "content": "Body",
        }
    ]

    def _fetch(ticker: str, *, session=None, lookback_days: int = 30, max_headlines: int = 40):
        if session is None:
            raise OSError("SSL certificate problem")
        return good_row

    strategies = [("yfinance_default", None), ("curl_cffi_no_verify", MagicMock())]

    with (
        patch("src.ingestion.adapters.yfinance_news.fetch_ticker_news", side_effect=_fetch),
        patch(
            "src.ingestion.adapters.yfinance_news.yfinance_news_strategies",
            return_value=strategies,
        ),
    ):
        rows, active = fetch_ticker_news_resilient("NVDA")

    assert rows == good_row
    assert active is not None
    assert active[0] == "curl_cffi_no_verify"


def test_load_yfinance_news_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    today = date.today().isoformat()
    monkeypatch.setattr(
        "src.ingestion.adapters.yfinance_news.load_app_config",
        lambda: MagicMock(market=MagicMock(tickers=["AAPL"])),
    )
    monkeypatch.setattr(
        "src.ingestion.adapters.yfinance_news.fetch_ticker_news_resilient",
        lambda *_a, **_k: (
            [
                {
                    "date": today,
                    "ticker": "AAPL",
                    "source": "Reuters",
                    "title": "Test headline",
                    "content": "Body",
                }
            ],
            ("curl_cffi_no_verify", MagicMock()),
        ),
    )

    df = load_yfinance_news()
    assert isinstance(df, pl.DataFrame)
    assert list(df.columns) == RAW_NEWS_COLUMNS
    assert df.height == 1
