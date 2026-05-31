"""Partitioned parquet write helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars import PartitionBy

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
    """Write market LazyFrame to Hive-partitioned parquet via lazy sink.

    Uses ``LazyFrame.sink_parquet(PartitionBy(...))`` (Polars >= 1.20).
    Falls back to collect + write_parquet on older Polars builds.
    """
    ensure_dir(output_dir)
    lf = add_partition_columns(lf)
    try:
        lf.sink_parquet(
            PartitionBy(
                str(output_dir),
                key=["year", "month"],
                include_key=True,
            ),
            compression=compression,
            mkdir=True,
        )
    except (TypeError, ValueError, ImportError):
        df = lf.collect()
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
