"""Polars lazy I/O helpers (sink_parquet)."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def sink_lazy_parquet(
    lf: pl.LazyFrame,
    path: Path | str,
    *,
    compression: str = "zstd",
) -> None:
    """Stream a LazyFrame to a single parquet file without an intermediate collect."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(str(out), compression=compression, mkdir=True)
