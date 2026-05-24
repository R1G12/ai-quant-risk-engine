"""Sample / synthetic market data adapter for CI and offline runs."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.utils.config import MarketConfig
from src.utils.paths import EXTERNAL_SAMPLE_MARKET_DIR, ensure_dir


def _generate_synthetic_ohlcv(cfg: MarketConfig) -> pl.LazyFrame:
    """Generate deterministic OHLCV for configured tickers and date range."""
    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    rows: list[dict] = []
    day = start
    seed = 42
    while day <= end:
        if day.weekday() < 5:
            for i, ticker in enumerate(cfg.tickers):
                base = 100.0 + i * 25.0 + (day.toordinal() % 17)
                noise = ((seed + i + day.day) % 7) * 0.3
                close = base + noise
                open_ = close - 0.5
                high = close + 1.0
                low = close - 1.0
                volume = 1_000_000 + (i * 100_000) + day.day * 1000
                rows.append(
                    {
                        "timestamp": day.isoformat(),
                        "ticker": ticker,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": float(volume),
                        "source": "sample",
                    }
                )
        day += timedelta(days=1)

    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC"),
    )
    return df.lazy()


def load_sample_market(cfg: MarketConfig) -> pl.LazyFrame:
    """Load bundled sample parquet if present, else generate synthetic data."""
    ensure_dir(EXTERNAL_SAMPLE_MARKET_DIR)
    parquet_files = list(EXTERNAL_SAMPLE_MARKET_DIR.glob("**/*.parquet"))
    if parquet_files:
        return pl.scan_parquet(str(EXTERNAL_SAMPLE_MARKET_DIR / "**/*.parquet"))
    return _generate_synthetic_ohlcv(cfg)


def ensure_external_sample_written(cfg: MarketConfig) -> None:
    """Write bundled sample parquet to external dir for reproducible CI."""
    ensure_dir(EXTERNAL_SAMPLE_MARKET_DIR)
    if list(EXTERNAL_SAMPLE_MARKET_DIR.glob("**/*.parquet")):
        return
    lf = _generate_synthetic_ohlcv(cfg)
    from src.market.partitions import sink_partitioned_market

    sink_partitioned_market(lf, EXTERNAL_SAMPLE_MARKET_DIR, compression=cfg.compression)
