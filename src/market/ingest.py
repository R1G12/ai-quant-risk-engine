"""Market data ingestion stage."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.market.adapters.sample import ensure_external_sample_written, load_sample_market
from src.market.adapters.yfinance import load_yfinance_market
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.paths import METRICS_DIR, RAW_MARKET_DIR, ensure_dir

LOGGER = get_logger(__name__)


def ingest_market_data(output_dir: Path | None = None) -> Path:
    """Ingest market OHLCV into raw parquet landing zone."""
    app = load_app_config()
    cfg = app.market
    out = output_dir or RAW_MARKET_DIR
    ensure_dir(out)

    LOGGER.info("Ingesting market data", extra={"source": cfg.source})

    if cfg.source == "yfinance":
        lf = load_yfinance_market(cfg)
    else:
        ensure_external_sample_written(cfg)
        lf = load_sample_market(cfg)

    raw_path = out / "market_raw.parquet"
    sink_lazy_parquet(lf, raw_path, compression=cfg.compression)
    LOGGER.info("Raw market data written", extra={"path": str(raw_path)})

    metrics_dir = METRICS_DIR / "market_ingest"
    log_stage_metrics(
        metrics_dir,
        {"source": cfg.source, "tickers": len(cfg.tickers)},
        lazy_frame=pl.scan_parquet(raw_path),
    )
    return out


if __name__ == "__main__":
    ingest_market_data()
