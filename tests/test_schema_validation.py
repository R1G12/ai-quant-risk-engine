"""Schema validation tests."""

import polars as pl

from src.validation.schema import validate_columns, validate_schema_file


def test_validate_market_schema() -> None:
    df = pl.DataFrame(
        {
            "timestamp": ["2024-01-02"],
            "ticker": ["AAPL"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC"))
    validate_schema_file(df, "market")


def test_validate_columns_raises() -> None:
    df = pl.DataFrame({"a": [1]})
    try:
        validate_columns(df, ["timestamp", "ticker"])
        raised = False
    except ValueError:
        raised = True
    assert raised
