"""Tests for sentiment feature aggregation (sample vs yfinance date dtypes)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.features.sentiment_agg import aggregate_sentiment_daily, map_sentiment_to_tickers
from src.utils.config import load_app_config


def _sentiment_lazy(tmp_path: Path, date_series: pl.Series) -> pl.LazyFrame:
    path = tmp_path / "sentiment.parquet"
    pl.DataFrame(
        {
            "date": date_series,
            "ticker": ["SMH", "SMH"],
            "source": ["yfinance", "yfinance"],
            "text": ["headline a", "headline b"],
            "sentiment_label": ["positive", "negative"],
            "sentiment_score": [0.9, 0.8],
        }
    ).write_parquet(path)
    return pl.scan_parquet(path)


@pytest.mark.parametrize(
    "date_series",
    [
        pl.Series("date", ["2026-06-01", "2026-06-01"], dtype=pl.Utf8),
        pl.Series("date", ["2026-06-01", "2026-06-01"], dtype=pl.Date),
    ],
    ids=["utf8_date", "pl_date"],
)
def test_map_and_aggregate_sentiment_date_dtypes(
    tmp_path: Path, date_series: pl.Series,
) -> None:
    app = load_app_config()
    lf = _sentiment_lazy(tmp_path, date_series).pipe(map_sentiment_to_tickers, app).pipe(
        aggregate_sentiment_daily
    )
    out = lf.collect()
    assert out.height >= 1
    assert "timestamp" in out.columns
    assert out["ticker"].to_list() == ["SMH"]
    assert out["article_count"].sum() == 2
