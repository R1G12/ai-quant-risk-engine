"""DVC stage: generate_correlations."""

from __future__ import annotations

import polars as pl

from src.features.wide_returns import load_returns_wide
from src.risk.correlations.covariance import (
    covariance_to_long,
    ledoit_wolf_shrinkage,
    sample_covariance_matrix,
)
from src.risk.correlations.rolling_corr import rolling_pairwise_corr_at_date
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.holdings import load_weights
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, RISK_CORRELATIONS_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_CORRELATIONS_DIR)
    tickers = list(load_weights(app).keys())
    window = app.features.correlation_window

    wide = load_returns_wide(tickers)

    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)

    ts = wide["timestamp"][-1] if wide.height else None
    cov_long = covariance_to_long(ts, tickers, cov, metric="cov")
    corr_long = rolling_pairwise_corr_at_date(wide, tickers, window)

    combined = pl.concat([cov_long, corr_long], how="diagonal_relaxed")
    write_single_parquet(
        combined,
        RISK_CORRELATIONS_DIR / "correlations_latest.parquet",
        app.market.compression,
    )

    LOGGER.info("Correlation/covariance written", extra={"path": str(RISK_CORRELATIONS_DIR)})
    log_stage_metrics(METRICS_DIR / "risk_correlations", {"pairs": combined.height})


if __name__ == "__main__":
    run()
