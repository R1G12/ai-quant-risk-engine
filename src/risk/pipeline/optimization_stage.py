"""DVC stage: optimize_portfolios."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.features.wide_returns import load_returns_wide
from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix
from src.risk.optimization.constraints import PortfolioConstraints
from src.risk.optimization.markowitz import max_sharpe_weights, min_variance_weights, portfolio_stats
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.holdings import load_weights
from src.risk.portfolio.returns import build_portfolio_returns
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import (
    METRICS_DIR,
    RISK_DATASET_PATH,
    RISK_OPT_METADATA_PATH,
    RISK_OPT_WEIGHTS_PATH,
    RISK_OPTIMIZATION_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def _mean_returns_with_sentiment(app, tickers: list[str]) -> np.ndarray:
    df = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .group_by("ticker")
        .agg(
            pl.col("returns").mean().alias("mu"),
            pl.col("bullish_ratio").mean().alias("sent"),
        )
        .collect()
    )
    scale = app.risk.portfolio.sentiment_return_scale
    mus = []
    for t in tickers:
        row = df.filter(pl.col("ticker") == t)
        if row.height == 0:
            mus.append(0.0)
            continue
        mu = float(row["mu"][0])
        if app.risk.optimization.use_sentiment_adjustment and row["sent"][0] is not None:
            mu += scale * (float(row["sent"][0]) - 0.5)
        mus.append(mu)
    return np.array(mus)


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_OPTIMIZATION_DIR)

    tickers = list(load_weights(app).keys())
    port = build_portfolio_returns(app)
    wide = load_returns_wide(tickers)

    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)

    mean_r = _mean_returns_with_sentiment(app, tickers)
    opt_cfg = app.risk.optimization
    cons = PortfolioConstraints(
        long_only=opt_cfg.long_only,
        min_weight=opt_cfg.min_weight,
        max_weight=opt_cfg.max_weight,
        leverage_cap=opt_cfg.leverage_cap,
    )

    w_min = min_variance_weights(cov, cons, len(tickers))
    w_sharpe, converged = max_sharpe_weights(
        mean_r, cov, cons, risk_free=app.features.risk_free_rate
    )

    weight_rows = []
    stat_rows = []
    for label, w in [("min_variance", w_min), ("max_sharpe", w_sharpe)]:
        r, v, s = portfolio_stats(w, mean_r, cov)
        for t, wi in zip(tickers, w, strict=False):
            weight_rows.append({"portfolio": label, "asset": t, "weight": float(wi)})
        stat_rows.append(
            {
                "portfolio": label,
                "expected_return": r,
                "volatility": v,
                "sharpe": s,
            }
        )

    weights_df = pl.DataFrame(weight_rows)
    stats_df = pl.DataFrame(stat_rows)
    write_single_parquet(stats_df, RISK_OPTIMIZATION_DIR / "optimization_stats.parquet", app.market.compression)
    write_single_parquet(weights_df, RISK_OPT_WEIGHTS_PATH, app.market.compression)

    meta = {
        "converged": converged,
        "shrinkage": opt_cfg.shrinkage,
        "tickers": tickers,
    }
    RISK_OPT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RISK_OPT_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_stage_metrics(
        METRICS_DIR / "risk_optimization",
        {"converged": int(converged), "portfolio_vol": float(port["portfolio_return"].std())},
    )
    LOGGER.info("Optimization complete", extra={"converged": converged})


if __name__ == "__main__":
    run()
