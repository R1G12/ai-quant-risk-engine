"""Tests for news ingestion."""

from pathlib import Path

import polars as pl

from src.ingestion.ingest import fetch_news


def test_fetch_news_writes_csv_with_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "news.csv"
    path = fetch_news(output_path=out, source="sample")

    assert path == out
    assert path.is_file()

    df = pl.read_csv(path)
    assert df.height == 3
    assert set(df.columns) == {"date", "source", "title", "content"}
