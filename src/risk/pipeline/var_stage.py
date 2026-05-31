"""DVC stage: generate_var_metrics."""

from __future__ import annotations

import polars as pl

from src.risk.metrics.tail_risk import compute_var_table
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.returns import build_portfolio_returns
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, RISK_VAR_DIR, ensure_dir
from src.validation.schema import validate_columns

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_VAR_DIR)

    port = build_portfolio_returns(app)
    var_df = compute_var_table(port["portfolio_return"], app.risk.var).with_columns(
        pl.lit("portfolio").alias("scope")
    )
    validate_columns(var_df, ["method", "confidence", "var"])
    write_single_parquet(var_df, RISK_VAR_DIR / "var_metrics.parquet", app.market.compression)
    LOGGER.info("VaR metrics written", extra={"path": str(RISK_VAR_DIR)})
    log_stage_metrics(
        METRICS_DIR / "risk_var",
        {"var_rows": var_df.height},
    )


if __name__ == "__main__":
    run()
