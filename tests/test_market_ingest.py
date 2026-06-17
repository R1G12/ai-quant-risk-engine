"""Market ingestion and cleaning tests."""

from pathlib import Path

import polars as pl
import pytest

from src.market.clean import clean_market_data
from src.market.ingest import ingest_market_data
from tests.helpers.platform_fixtures import CI_SAMPLE_TICKERS


@pytest.fixture
def sample_market_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate sample ingest from synced params.yaml / live run.yaml tickers."""
    monkeypatch.setenv("MARKET_SOURCE", "sample")
    monkeypatch.setenv("RUN_PROFILE", "configs/run.ci.yaml")
    monkeypatch.setenv("MARKET_PIN_DATES", "1")


def test_ingest_and_clean_sample_pipeline(
    tmp_path: Path, sample_market_env: None,
) -> None:
    raw = tmp_path / "raw" / "market"
    processed = tmp_path / "processed" / "market"

    ingest_market_data(output_dir=raw)
    assert (raw / "market_raw.parquet").is_file()

    clean_market_data(input_dir=raw, output_dir=processed)
    lf = pl.scan_parquet(str(processed / "**" / "*.parquet"))
    df = lf.select("ticker", "close").collect()
    assert df.height > 0
    tickers = set(df["ticker"].unique().to_list())
    assert tickers.issubset(set(CI_SAMPLE_TICKERS))
    assert len(tickers) >= 2
    assert "AAPL" in tickers
