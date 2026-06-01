"""Tests for news ingestion adapters."""

from __future__ import annotations

import os

import polars as pl
import pytest

from src.ingestion.adapters import get_news_source, load_news
from src.ingestion.adapters.sample import load_sample_news


def test_load_sample_news_schema() -> None:
    df = load_sample_news()
    assert df.height == 3
    assert set(df.columns) == {"date", "ticker", "source", "title", "content"}
    assert df["ticker"].to_list() == ["AAPL", "XOM", "GS"]


def test_yfinance_adapter_registered() -> None:
    from src.ingestion.adapters import _ADAPTERS

    assert "yfinance" in _ADAPTERS


def test_load_news_sample_adapter() -> None:
    df = load_news("sample")
    assert isinstance(df, pl.DataFrame)
    assert df.height == 3


def test_get_news_source_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_SOURCE", "sample")
    assert get_news_source() == "sample"


def test_load_news_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown news source"):
        load_news("unknown_api")
