# Polars Guidelines

Production code in `src/` must follow these rules.

## Core principles

1. **Lazy-first** — build `LazyFrame` pipelines; materialize only when required.
2. **Parquet-only** — no CSV in production paths (Phase 1 news CSV migration is optional).
3. **No pandas** in `src/` except the documented yfinance boundary in `src/market/adapters/yfinance.py`.
4. **Vectorized only** — no Python loops over rows; use `over("ticker")` for per-symbol logic.
5. **Minimize `collect()`** — allowed for schema checks, small samples, and DVCLive row counts.

## I/O patterns

```python
# Read partitioned lake
lf = pl.scan_parquet("data/processed/market/year=*/month=*/*.parquet")

# Write single parquet (streaming)
from src.utils.io import sink_lazy_parquet
sink_lazy_parquet(lf, "data/features/returns/returns.parquet", compression="zstd")

# Write Hive-partitioned market data (streaming)
from polars import PartitionBy
lf.sink_parquet(
    PartitionBy("data/processed/market", key=["year", "month"], include_key=True),
    compression="zstd",
    mkdir=True,
)
```

Prefer `scan_parquet()` over `read_parquet()` for multi-file inputs.

Project helper: `src/market/partitions.py` → `sink_partitioned_market()` wraps `PartitionBy` (requires **Polars >= 1.20**).

## Feature engineering

- Compose with `.pipe()`:

```python
lf = (
    pl.scan_parquet(path)
    .pipe(add_returns)
    .pipe(add_volatility, window=21)
)
```

- Per-ticker windows: `.over("ticker")`
- Returns: `pct_change()`, `log1p` patterns — not iterative row access

## Anti-patterns

| Avoid | Use instead |
|-------|-------------|
| `for row in df.iter_rows()` | Vectorized expressions |
| `map_elements` on large columns | Native Polars expressions |
| `read_parquet` on many files | `scan_parquet` + glob |
| `collect()` then `write_parquet` on large outputs | `sink_lazy_parquet` or `PartitionBy` sink |
| pandas in `src/` | Polars LazyFrame |

## Approved exception

`src/market/adapters/yfinance.py` may call `yfinance` (pandas internally) and must convert to Polars immediately via `pl.from_pandas()` at the module boundary.
