"""DVC: run_backtests."""

from __future__ import annotations

import polars as pl

from src.backtesting.engine.walk_forward import run_historical_backtest
from src.backtesting.metrics.performance import max_drawdown, sharpe_ratio
from src.features.wide_returns import load_returns_wide
from src.risk.portfolio.holdings import load_weights
from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.io import sink_lazy_parquet
from src.utils.logger import get_logger
from src.utils.paths import (
    EXPERIMENTS_BACKTESTS_DIR,
    METRICS_DIR,
    RESEARCH_BACKTESTS_DIR,
    RISK_OPT_WEIGHTS_PATH,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def _load_weights(app) -> dict[str, float]:
    if RISK_OPT_WEIGHTS_PATH.is_file():
        df = pl.read_parquet(RISK_OPT_WEIGHTS_PATH)
        df = df.filter(pl.col("asset") != "_stats_")
        if app.research.backtest.weight_source:
            sub = df.filter(pl.col("portfolio") == app.research.backtest.weight_source)
            if sub.height:
                return dict(zip(sub["asset"].to_list(), sub["weight"].to_list(), strict=False))
    return load_weights(app)


def run() -> None:
    app = load_app_config()
    exp_id = app.research.meta.experiment_id
    weights = _load_weights(app)
    tickers = list(weights.keys())

    wide = load_returns_wide(tickers)

    dupes = wide.group_by("timestamp").len().filter(pl.col("len") > 1)
    if dupes.height:
        LOGGER.warning("Duplicate timestamps remain after dedupe", extra={"count": dupes.height})

    equity, rolling, trades = run_historical_backtest(wide, weights, app)
    out_dir = RESEARCH_BACKTESTS_DIR / f"experiment_id={exp_id}"
    ensure_dir(out_dir)
    sink_lazy_parquet(equity, out_dir / "equity_curve.parquet", compression=app.market.compression)
    sink_lazy_parquet(rolling, out_dir / "rolling_metrics.parquet", compression=app.market.compression)
    sink_lazy_parquet(trades, out_dir / "trades.parquet", compression=app.market.compression)

    sharpe = sharpe_ratio(equity["portfolio_return"], app.features.risk_free_rate, app.features.annualization_factor)
    mdd = max_drawdown(equity["equity"])

    log_research_metrics(
        METRICS_DIR / "research_backtests",
        {"sharpe": sharpe, "max_drawdown": mdd, "n_days": equity.height},
        manifest_path=EXPERIMENTS_BACKTESTS_DIR / exp_id / "manifest.yaml",
        experiment_id=exp_id,
        stage="run_backtests",
        params={"weight_source": app.research.backtest.weight_source},
        outputs=[str(out_dir)],
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Backtest complete", extra={"sharpe": sharpe, "max_drawdown": mdd})


if __name__ == "__main__":
    run()
