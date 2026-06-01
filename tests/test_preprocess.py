"""Tests for preprocessing."""

from pathlib import Path

import polars as pl

from src.ingestion.ingest import fetch_news
from src.preprocessing.preprocess import TEXT_COLUMN, preprocess_news


def test_preprocess_news_produces_text_column(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "news.csv"
    raw.parent.mkdir(parents=True)
    fetch_news(output_path=raw, source="sample")

    out = tmp_path / "processed" / "news.parquet"
    preprocess_news(input_path=raw, output_path=out)

    df = pl.read_parquet(out)
    assert TEXT_COLUMN in df.columns
    assert df.height == 3
    assert df[TEXT_COLUMN].null_count() == 0
    assert (df[TEXT_COLUMN].str.strip_chars() == df[TEXT_COLUMN]).all()
