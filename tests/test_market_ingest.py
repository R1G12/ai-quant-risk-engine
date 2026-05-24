"""Market ingestion and cleaning tests."""

from pathlib import Path

import polars as pl

from src.market.clean import clean_market_data
from src.market.ingest import ingest_market_data


def test_ingest_and_clean_sample_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MARKET_SOURCE", "sample")
    raw = tmp_path / "raw" / "market"
    processed = tmp_path / "processed" / "market"

    ingest_market_data(output_dir=raw)
    assert (raw / "market_raw.parquet").is_file()

    clean_market_data(input_dir=raw, output_dir=processed)
    lf = pl.scan_parquet(str(processed / "**" / "*.parquet"))
    df = lf.select("ticker", "close").collect()
    assert df.height > 0
    assert "AAPL" in df["ticker"].to_list()
