"""Tests for Signals dashboard data loaders."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.dashboards.core.signals_loaders import load_finbert_window
from src.utils.config import load_app_config


@pytest.fixture
def sentiment_parquet(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent_path = tmp_path / "sentiment.parquet"
    monkeypatch.setattr(
        "src.portfolio.sentiment_sides.PROCESSED_SENTIMENT_PATH",
        sent_path,
    )
    monkeypatch.setattr(
        "src.dashboards.core.signals_loaders.PROCESSED_SENTIMENT_PATH",
        sent_path,
    )
    end = date.today()
    rows = []
    for i in range(5):
        d = (end - timedelta(days=i)).isoformat()
        rows.append(
            {
                "date": d,
                "source": "Reuters" if i % 2 else "Bloomberg",
                "text": "headline",
                "sentiment_label": "positive",
                "sentiment_score": 0.9,
            }
        )
    pl.DataFrame(rows).write_parquet(tmp_path / "sentiment.parquet")


def test_load_finbert_window_maps_source(monkeypatch: pytest.MonkeyPatch, sentiment_parquet: None) -> None:
    monkeypatch.setattr(
        "src.dashboards.core.signals_loaders.load_weights",
        lambda _app: {"AAPL": 0.5, "XOM": 0.5},
    )
    app = load_app_config()
    summary, daily = load_finbert_window(app, window_days=30, tickers=["AAPL", "XOM"])
    assert summary is not None
    assert daily is not None
    assert "XOM" in summary["ticker"].to_list()


def test_load_finbert_window_prefers_ticker_column(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_path = tmp_path / "sentiment.parquet"
    monkeypatch.setattr(
        "src.portfolio.sentiment_sides.PROCESSED_SENTIMENT_PATH",
        sent_path,
    )
    monkeypatch.setattr(
        "src.dashboards.core.signals_loaders.PROCESSED_SENTIMENT_PATH",
        sent_path,
    )
    end = date.today()
    rows = [
        {
            "date": end.isoformat(),
            "ticker": "NVDA",
            "source": "ignored",
            "text": "chip rally",
            "sentiment_label": "positive",
            "sentiment_score": 0.85,
        }
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / "sentiment.parquet")
    monkeypatch.setattr(
        "src.dashboards.core.signals_loaders.load_weights",
        lambda _app: {"NVDA": 1.0},
    )
    app = load_app_config()
    summary, daily = load_finbert_window(app, window_days=30, tickers=["NVDA"])
    assert summary is not None
    assert daily is not None
    assert summary["ticker"].to_list() == ["NVDA"]
