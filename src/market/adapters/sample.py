"""Sample / synthetic market data adapter for CI and offline runs."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.market.ticker_validation import (
    TickerFilterResult,
    format_skip_messages,
    require_min_tickers,
    validate_market_tickers,
)
from src.utils.config import MarketConfig
from src.utils.logger import get_logger
from src.utils.paths import EXTERNAL_SAMPLE_MARKET_DIR, ensure_dir

LOGGER = get_logger(__name__)


def _generate_synthetic_ohlcv(cfg: MarketConfig, tickers: list[str]) -> pl.LazyFrame:
    """Generate deterministic OHLCV for explicit ticker list (CI bootstrap only)."""
    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    rows: list[dict] = []
    day = start
    seed = 42
    while day <= end:
        if day.weekday() < 5:
            for i, ticker in enumerate(tickers):
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
    LOGGER.warning(
        "Generated synthetic sample OHLCV for %d ticker(s): %s",
        len(tickers),
        tickers,
    )
    return df.lazy()


def filter_sample_tickers(tickers: list[str]) -> TickerFilterResult:
    """Return tickers available in bundled sample data (no synthetic for missing)."""
    return validate_market_tickers(tickers, "sample")


def load_sample_market(cfg: MarketConfig) -> pl.LazyFrame:
    """Load bundled sample parquet for validated tickers only."""
    result = filter_sample_tickers(cfg.tickers)
    for line in format_skip_messages(result):
        LOGGER.warning(line)
    require_min_tickers(result, context="sample market ingest")

    ensure_dir(EXTERNAL_SAMPLE_MARKET_DIR)
    parquet_files = list(EXTERNAL_SAMPLE_MARKET_DIR.glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(
            "No bundled sample market data found. Run CI bootstrap or set MARKET_SOURCE=yfinance."
        )

    return (
        pl.scan_parquet(str(EXTERNAL_SAMPLE_MARKET_DIR / "**/*.parquet"))
        .filter(pl.col("ticker").is_in(result.valid))
    )


def ensure_external_sample_written(cfg: MarketConfig) -> None:
    """Write bundled sample parquet to external dir for reproducible CI (first run only)."""
    ensure_dir(EXTERNAL_SAMPLE_MARKET_DIR)
    if list(EXTERNAL_SAMPLE_MARKET_DIR.glob("**/*.parquet")):
        return
    # First-time CI bootstrap: synthesize only the configured params tickers once.
    lf = _generate_synthetic_ohlcv(cfg, list(cfg.tickers))
    from src.market.partitions import sink_partitioned_market

    sink_partitioned_market(lf, EXTERNAL_SAMPLE_MARKET_DIR, compression=cfg.compression)
