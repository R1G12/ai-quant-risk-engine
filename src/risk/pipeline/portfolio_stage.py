"""DVC stage: generate_portfolio_metrics."""

from __future__ import annotations

import polars as pl

from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.analytics import add_portfolio_analytics, max_drawdown
from src.risk.portfolio.exposures import load_exposure_table
from src.risk.portfolio.holdings import ensure_holdings
from src.risk.portfolio.returns import build_portfolio_returns
from src.risk.regimes.hmm import fit_hmm_regimes
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import (
    METRICS_DIR,
    RISK_PORTFOLIO_DIR,
    RISK_PORTFOLIO_METRICS_PATH,
    RISK_PORTFOLIO_RETURNS_PATH,
    RISK_REGIMES_PATH,
    ensure_dir,
)
from src.validation.schema import validate_schema_file

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_holdings(app)
    ensure_dir(RISK_PORTFOLIO_DIR)

    port = build_portfolio_returns(app)
    metrics = add_portfolio_analytics(port.lazy(), app).collect()
    validate_schema_file(metrics, "risk_portfolio")

    write_single_parquet(port, RISK_PORTFOLIO_RETURNS_PATH, app.market.compression)
    write_single_parquet(metrics, RISK_PORTFOLIO_METRICS_PATH, app.market.compression)

    valid = port.filter(pl.col("portfolio_return").is_not_null())
    regimes = fit_hmm_regimes(valid["portfolio_return"], app.risk, seed=app.risk.var.seed)
    if regimes.height == valid.height:
        regimes = regimes.with_columns(valid["timestamp"])
        write_single_parquet(regimes, RISK_REGIMES_PATH, app.market.compression)

    exposures = load_exposure_table(app)
    write_single_parquet(exposures, RISK_PORTFOLIO_DIR / "exposures.parquet", app.market.compression)

    mdd = max_drawdown(port["portfolio_return"])
    LOGGER.info("Portfolio metrics written", extra={"max_drawdown": mdd})
    log_stage_metrics(
        METRICS_DIR / "risk_portfolio",
        {"max_drawdown": mdd, "sharpe_last": float(metrics["rolling_sharpe"][-1])},
    )


if __name__ == "__main__":
    run()
