"""Optional benchmark: scan_parquet vs eager read (Phase 2 stretch)."""

from __future__ import annotations

import time

import polars as pl

from src.utils.paths import PROCESSED_MARKET_DIR


def main() -> None:
    glob = str(PROCESSED_MARKET_DIR / "**" / "*.parquet")
    t0 = time.perf_counter()
    eager = pl.read_parquet(glob)
    eager_rows = eager.height
    t1 = time.perf_counter()
    lazy = pl.scan_parquet(glob).select(pl.len()).collect()
    t2 = time.perf_counter()
    print(f"eager: {eager_rows} rows in {t1 - t0:.3f}s")
    print(f"lazy count: {lazy.item()} rows in {t2 - t1:.3f}s")


if __name__ == "__main__":
    main()
