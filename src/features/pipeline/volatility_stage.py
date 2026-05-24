"""DVC stage: generate_volatility_features."""

from __future__ import annotations

import polars as pl

from src.features.liquidity import add_liquidity_features
from src.features.risk_ratios import add_rolling_sharpe
from src.features.volatility import add_volatility_features
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.paths import FEATURES_RETURNS_DIR, FEATURES_VOLATILITY_DIR, METRICS_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(FEATURES_VOLATILITY_DIR)
    lf = (
        pl.scan_parquet(FEATURES_RETURNS_DIR / "returns.parquet")
        .pipe(add_volatility_features, app.features)
        .pipe(add_rolling_sharpe, app.features)
        .pipe(add_liquidity_features, app.features.volatility_window)
    )
    out = FEATURES_VOLATILITY_DIR / "volatility.parquet"
    sink_lazy_parquet(lf, out, compression=app.market.compression)
    LOGGER.info("Volatility features written", extra={"path": str(out)})
    log_stage_metrics(METRICS_DIR / "features_volatility", {}, lazy_frame=lf)


if __name__ == "__main__":
    run()
