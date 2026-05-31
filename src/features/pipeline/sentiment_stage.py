"""DVC stage: generate_sentiment_features."""

from __future__ import annotations

import polars as pl

from src.features.sentiment_agg import aggregate_sentiment_daily, map_sentiment_to_tickers
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.paths import (
    FEATURES_SENTIMENT_AGG_DIR,
    METRICS_DIR,
    PROCESSED_SENTIMENT_PATH,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(FEATURES_SENTIMENT_AGG_DIR)
    if not PROCESSED_SENTIMENT_PATH.is_file():
        LOGGER.warning("Sentiment input missing; writing empty aggregation")
        empty = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime(time_zone="UTC"),
                "ticker": pl.Utf8,
                "confidence": pl.Float64,
                "sentiment": pl.Utf8,
                "bullish_ratio": pl.Float64,
                "article_count": pl.UInt32,
            }
        )
        out = FEATURES_SENTIMENT_AGG_DIR / "sentiment_agg.parquet"
        empty.write_parquet(out)
        return

    lf = (
        pl.scan_parquet(PROCESSED_SENTIMENT_PATH)
        .pipe(map_sentiment_to_tickers, app)
        .pipe(aggregate_sentiment_daily)
    )
    out = FEATURES_SENTIMENT_AGG_DIR / "sentiment_agg.parquet"
    sink_lazy_parquet(lf, out, compression=app.market.compression)
    LOGGER.info("Sentiment features written", extra={"path": str(out)})
    log_stage_metrics(METRICS_DIR / "features_sentiment", {}, lazy_frame=lf)


if __name__ == "__main__":
    run()
