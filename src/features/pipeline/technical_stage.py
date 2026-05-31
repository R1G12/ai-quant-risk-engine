"""DVC stage: generate_technical_features."""

from __future__ import annotations

import polars as pl

from src.features.correlation import add_rolling_correlation_to_index
from src.features.technical import add_technical_features
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.paths import FEATURES_RETURNS_DIR, FEATURES_TECHNICAL_DIR, METRICS_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(FEATURES_TECHNICAL_DIR)
    lf = pl.scan_parquet(FEATURES_RETURNS_DIR / "returns.parquet").pipe(
        add_technical_features, app.features
    )
    if app.market.tickers and "AAPL" in app.market.tickers:
        lf = lf.pipe(add_rolling_correlation_to_index, app.features, "AAPL")
    out = FEATURES_TECHNICAL_DIR / "technical.parquet"
    sink_lazy_parquet(lf, out, compression=app.market.compression)
    LOGGER.info("Technical features written", extra={"path": str(out)})
    log_stage_metrics(METRICS_DIR / "features_technical", {}, lazy_frame=lf)


if __name__ == "__main__":
    run()
