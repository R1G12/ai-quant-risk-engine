"""DVC stage: generate_cvar_metrics."""

from __future__ import annotations

from src.risk.metrics.tail_risk import compute_cvar_table
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.returns import build_portfolio_returns
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, RISK_CVAR_DIR, ensure_dir
from src.validation.schema import validate_columns

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_CVAR_DIR)

    port = build_portfolio_returns(app)
    cvar_df = compute_cvar_table(port["portfolio_return"], app.risk.var)
    validate_columns(cvar_df, ["method", "confidence", "cvar"])
    write_single_parquet(cvar_df, RISK_CVAR_DIR / "cvar_metrics.parquet", app.market.compression)
    LOGGER.info("CVaR metrics written", extra={"path": str(RISK_CVAR_DIR)})
    log_stage_metrics(METRICS_DIR / "risk_cvar", {"cvar_rows": cvar_df.height})


if __name__ == "__main__":
    run()
