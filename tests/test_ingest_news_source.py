"""Regression tests for news source resolution and stale sample data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.ingestion.adapters import get_news_source
from src.ingestion.news_refresh import holdings_missing_from_news
from src.utils.config import news_source_for_run_mode


def test_get_news_source_defaults_to_sample_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWS_SOURCE", raising=False)
    assert get_news_source() == "sample"


def test_live_mode_expects_yfinance_news_source() -> None:
    assert news_source_for_run_mode("live") == "yfinance"


def test_holdings_missing_from_sample_news(tmp_path: Path) -> None:
    news = tmp_path / "news.parquet"
    pl.DataFrame(
        {
            "date": ["2026-01-01"],
            "ticker": ["AAPL"],
            "source": ["Bloomberg"],
            "title": ["t"],
            "content": ["c"],
        }
    ).write_parquet(news)
    assert holdings_missing_from_news(["SMH", "MSFT"], news)


def test_holdings_present_in_yfinance_news(tmp_path: Path) -> None:
    news = tmp_path / "news.parquet"
    pl.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-01"],
            "ticker": ["SMH", "MSFT"],
            "source": ["yfinance", "yfinance"],
            "title": ["a", "b"],
            "content": ["a", "b"],
        }
    ).write_parquet(news)
    assert not holdings_missing_from_news(["SMH", "MSFT"], news)
