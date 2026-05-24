"""Risk pipeline I/O helpers."""

from __future__ import annotations

import polars as pl

from src.market.partitions import add_partition_columns, sink_partitioned_market
from src.utils.io import sink_lazy_parquet


def add_scope_and_partitions(df: pl.DataFrame, scope: str) -> pl.LazyFrame:
    """Add scope label and year/month partition columns."""
    lf = df.lazy().with_columns(pl.lit(scope).alias("scope"))
    return add_partition_columns(lf)


def sink_risk_partitioned(lf: pl.LazyFrame, output_dir, compression: str = "zstd") -> None:
    """Write risk metrics with year/month partitions."""
    sink_partitioned_market(lf, output_dir, compression=compression)


def write_single_parquet(df: pl.DataFrame, path, compression: str = "zstd") -> None:
    sink_lazy_parquet(df.lazy(), path, compression=compression)
