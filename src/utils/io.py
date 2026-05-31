"""Polars lazy I/O helpers (sink_parquet)."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def sink_lazy_parquet(
    lf: pl.LazyFrame | pl.DataFrame,
    path: Path | str,
    *,
    compression: str = "zstd",
) -> None:
    """Stream a LazyFrame (or eager DataFrame) to parquet."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lf, pl.DataFrame):
        lf.write_parquet(out, compression=compression)
        return
    lf.sink_parquet(str(out), compression=compression, mkdir=True)
