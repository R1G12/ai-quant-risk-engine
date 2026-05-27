"""DVC stage: generate_efficient_frontier."""

from __future__ import annotations

from src.features.wide_returns import load_returns_wide
from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix
from src.risk.optimization.constraints import PortfolioConstraints
from src.risk.optimization.frontier import efficient_frontier
from src.risk.pipeline._io import write_single_parquet
from src.risk.pipeline.optimization_stage import _mean_returns_with_sentiment
from src.risk.portfolio.holdings import load_weights
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, RISK_FRONTIER_PATH, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    from src.utils.paths import RISK_FRONTIER_DIR

    ensure_dir(RISK_FRONTIER_DIR)

    tickers = list(load_weights(app).keys())
    wide = load_returns_wide(tickers)
    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)
    mean_r = _mean_returns_with_sentiment(app, tickers)

    opt = app.risk.optimization
    cons = PortfolioConstraints(
        long_only=opt.long_only,
        min_weight=opt.min_weight,
        max_weight=opt.max_weight,
        leverage_cap=opt.leverage_cap,
    )
    frontier = efficient_frontier(mean_r, cov, cons, n_points=opt.frontier_points)
    write_single_parquet(frontier, RISK_FRONTIER_PATH, app.market.compression)
    from src.analytics.risk_plots import write_risk_dashboard

    write_risk_dashboard()
    LOGGER.info("Efficient frontier written", extra={"path": str(RISK_FRONTIER_PATH)})
    log_stage_metrics(METRICS_DIR / "risk_frontier", {"points": frontier.height})


if __name__ == "__main__":
    run()
