"""Partitioned parquet write helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.paths import ensure_dir


def add_partition_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add year and month columns from timestamp."""
    return lf.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    )


def sink_partitioned_market(
    lf: pl.LazyFrame,
    output_dir: Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Write market LazyFrame to Hive-partitioned parquet.

    Uses ``DataFrame.write_parquet(partition_by=...)`` after collect for
    compatibility across Polars versions (``sink_parquet`` lacks partition_by).
    """
    ensure_dir(output_dir)
    df = add_partition_columns(lf).collect()
    df.write_parquet(
        str(output_dir),
        partition_by=["year", "month"],
        compression=compression,
    )
    return output_dir


def scan_market_dir(path: Path | str) -> pl.LazyFrame:
    """Scan partitioned or flat market parquet directory."""
    base = Path(path)
    return pl.scan_parquet(str(base / "**" / "*.parquet"))
