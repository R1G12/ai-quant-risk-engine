"""DVC stage: optimize_portfolios."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.features.wide_returns import load_returns_wide
from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix
from src.risk.optimization.constraints import PortfolioConstraints
from src.portfolio.sentiment_sides import effective_position_sides
from src.portfolio.weights import (
    optimize_max_sharpe_gross,
    optimize_partial_weights,
    resolve_weights,
)
from src.risk.optimization.markowitz import max_sharpe_weights, min_variance_weights, portfolio_stats
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.holdings import read_holdings_weights
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


def _run_portfolio_kw(app) -> dict:
    opt = app.risk.optimization
    if app.run is not None:
        p = app.run.portfolio
        return {
            "allow_shorts": p.allow_shorts,
            "max_gross_per_ticker": p.max_gross_per_ticker,
            "position_sides": p.position_sides or None,
            "manual_weights": p.manual_weights or None,
            "anchor_weights": p.anchor_weights or None,
            "risk_free": p.risk_free,
            "min_gross_divisor": p.min_gross_divisor,
        }
    return {
        "allow_shorts": opt.allow_shorts,
        "max_gross_per_ticker": opt.max_gross_per_ticker,
        "position_sides": None,
        "manual_weights": None,
        "anchor_weights": None,
        "risk_free": app.features.risk_free_rate,
        "min_gross_divisor": 5.0,
    }


def _max_sharpe_weights(
    app,
    tickers: list[str],
    mean_r: np.ndarray,
    cov: np.ndarray,
    cons: PortfolioConstraints,
) -> tuple[np.ndarray, bool]:
    opt = app.risk.optimization
    weighting = opt.weighting
    kw = _run_portfolio_kw(app)
    kw["position_sides"] = effective_position_sides(app, tickers, kw.get("position_sides"))

    if weighting == "optimised":
        return optimize_max_sharpe_gross(
            mean_r,
            cov,
            tickers,
            risk_free=kw["risk_free"],
            allow_shorts=kw["allow_shorts"],
            max_gross_per_ticker=kw["max_gross_per_ticker"],
            min_gross_divisor=kw["min_gross_divisor"],
            position_sides=kw["position_sides"],
        )

    if weighting == "partial":
        return optimize_partial_weights(
            mean_r,
            cov,
            tickers,
            kw["anchor_weights"] or {},
            risk_free=kw["risk_free"],
            allow_shorts=kw["allow_shorts"],
            max_gross_per_ticker=kw["max_gross_per_ticker"],
            min_gross_divisor=kw["min_gross_divisor"],
            position_sides=kw["position_sides"],
        )

    if weighting in ("equal", "manual"):
        w = resolve_weights(weighting, tickers, **kw)
        return w, True

    w, converged = max_sharpe_weights(mean_r, cov, cons, risk_free=app.features.risk_free_rate)
    return w, converged


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_OPTIMIZATION_DIR)

    tickers = list(read_holdings_weights(app).keys())
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
    w_sharpe, converged = _max_sharpe_weights(app, tickers, mean_r, cov, cons)

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
        "weighting": opt_cfg.weighting,
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
