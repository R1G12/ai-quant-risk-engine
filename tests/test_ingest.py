"""Tests for news ingestion."""

from pathlib import Path

import polars as pl

from src.ingestion.ingest import fetch_news
from src.ingestion.news_schema import RAW_NEWS_COLUMNS


def test_fetch_news_writes_parquet_with_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "news.parquet"
    path = fetch_news(output_path=out)

    assert path == out
    assert path.is_file()

    df = pl.read_parquet(path)
    assert df.height == 3
    assert list(df.columns) == RAW_NEWS_COLUMNS
