"""DVC stage: merge_features."""

from __future__ import annotations

import polars as pl

from src.features.merge import merge_feature_frames, write_metadata
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import (
    FEATURES_MERGED_DIR,
    FEATURES_RETURNS_DIR,
    FEATURES_SENTIMENT_AGG_DIR,
    FEATURES_TECHNICAL_DIR,
    FEATURES_VOLATILITY_DIR,
    METRICS_DIR,
    RISK_DATASET_METADATA_PATH,
    RISK_DATASET_PATH,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(FEATURES_MERGED_DIR)

    returns_lf = pl.scan_parquet(FEATURES_RETURNS_DIR / "returns.parquet")
    vol_lf = pl.scan_parquet(FEATURES_VOLATILITY_DIR / "volatility.parquet")
    tech_lf = pl.scan_parquet(FEATURES_TECHNICAL_DIR / "technical.parquet")
    sent_path = FEATURES_SENTIMENT_AGG_DIR / "sentiment_agg.parquet"
    sent_lf = pl.scan_parquet(sent_path) if sent_path.is_file() else pl.LazyFrame()

    merged = merge_feature_frames(returns_lf, vol_lf, tech_lf, sent_lf)
    merged.collect().write_parquet(RISK_DATASET_PATH, compression=app.market.compression)

    cols = merged.collect_schema().names()
    write_metadata(RISK_DATASET_METADATA_PATH, list(cols))
    LOGGER.info("Risk dataset written", extra={"path": str(RISK_DATASET_PATH)})
    log_stage_metrics(METRICS_DIR / "features_merge", {"feature_columns": len(cols)}, lazy_frame=merged)


if __name__ == "__main__":
    run()
