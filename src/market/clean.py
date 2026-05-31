"""Market data cleaning and partitioned lake write."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from src.market.partitions import sink_partitioned_market
from src.schemas.market import MARKET_CLEAN_COLUMNS
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, PROCESSED_MARKET_DIR, RAW_MARKET_DIR
from src.validation.schema import validate_columns

LOGGER = get_logger(__name__)


def clean_market_data(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Clean raw market data and write partitioned parquet lake."""
    app = load_app_config()
    cfg = app.market
    inp = input_dir or RAW_MARKET_DIR
    out = output_dir or PROCESSED_MARKET_DIR

    raw_path = inp / "market_raw.parquet"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw market file not found: {raw_path}")

    lf = (
        pl.scan_parquet(raw_path)
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
            pl.col("ticker").str.to_uppercase().alias("ticker"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        .filter(pl.col("close").is_not_null())
        .sort(["ticker", "timestamp"])
        .with_columns(pl.lit("clean").alias("source"))
        .with_columns(pl.lit(datetime.now(timezone.utc)).alias("ingested_at"))
    )

    sample = lf.select(list(MARKET_CLEAN_COLUMNS)).head(5).collect()
    validate_columns(sample, MARKET_CLEAN_COLUMNS)

    sink_partitioned_market(lf, out, compression=cfg.compression)
    LOGGER.info("Cleaned market data written", extra={"path": str(out)})

    metrics_dir = METRICS_DIR / "market_clean"
    log_stage_metrics(metrics_dir, {"compression": cfg.compression}, lazy_frame=lf)
    return out


if __name__ == "__main__":
    clean_market_data()
