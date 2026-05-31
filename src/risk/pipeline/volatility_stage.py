"""DVC stage: generate_volatility_metrics."""

from __future__ import annotations

import polars as pl

from src.risk.pipeline._io import sink_risk_partitioned
from src.risk.portfolio.holdings import ensure_holdings
from src.risk.portfolio.returns import build_portfolio_returns
from src.risk.volatility.ewma import add_ewma_volatility
from src.risk.volatility.garch import fit_garch_volatility
from src.risk.volatility.historical import add_rolling_volatility
from src.risk.volatility.regimes_simple import add_vol_regime_flags
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, RISK_VOLATILITY_DIR, ensure_dir
from src.validation.schema import validate_columns

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    ensure_holdings(app)
    ensure_dir(RISK_VOLATILITY_DIR)

    port = build_portfolio_returns(app).lazy()
    cfg = app.risk.volatility
    lf = (
        port.pipe(add_rolling_volatility, cfg)
        .pipe(add_ewma_volatility, cfg)
        .pipe(add_vol_regime_flags, cfg, vol_col="ewma_vol")
    )
    df = lf.collect()
    garch = fit_garch_volatility(df["portfolio_return"], cfg)
    df = df.with_columns(garch).with_columns(pl.lit("portfolio").alias("scope"))

    validate_columns(
        df,
        ["timestamp", "scope", "rolling_vol", "ewma_vol", "vol_regime_flag"],
        optional_columns=["garch_vol"],
    )
    sink_risk_partitioned(df.lazy(), RISK_VOLATILITY_DIR, compression=app.market.compression)
    LOGGER.info("Volatility metrics written", extra={"path": str(RISK_VOLATILITY_DIR)})
    log_stage_metrics(
        METRICS_DIR / "risk_volatility",
        {"ewma_span": cfg.ewma_span},
        lazy_frame=df.lazy(),
    )


if __name__ == "__main__":
    run()
