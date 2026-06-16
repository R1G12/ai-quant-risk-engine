"""DVC stage: optimize_portfolios."""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.features.wide_returns import load_returns_wide
from src.portfolio.expected_returns import build_expected_returns
from src.portfolio.sentiment_sides import effective_position_sides, finbert_scores_by_ticker
from src.portfolio.sentiment_tilt import apply_sentiment_magnitude_tilt
from src.portfolio.weights import (
    optimize_max_sharpe_gross,
    optimize_partial_weights,
    resolve_weights,
)
from src.risk.correlations.covariance import (
    align_optimization_tickers,
    ledoit_wolf_shrinkage,
    sample_covariance_matrix,
)
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
    RISK_OPT_METADATA_PATH,
    RISK_OPT_WEIGHTS_PATH,
    RISK_OPTIMIZATION_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def _run_portfolio_kw(app, tickers: list[str]) -> dict:
    opt = app.risk.optimization
    if app.run is not None:
        p = app.run.portfolio
        sides = effective_position_sides(app, tickers)
        if sides is None:
            sides = {k: v for k, v in (p.position_sides or {}).items() if k in tickers} or None
        return {
            "allow_shorts": p.allow_shorts,
            "max_gross_per_ticker": p.max_gross_per_ticker,
            "min_gross_divisor": p.min_gross_divisor,
            "position_sides": sides,
            "manual_weights": p.manual_weights or None,
            "anchor_weights": p.anchor_weights or None,
            "risk_free": p.risk_free,
        }
    return {
        "allow_shorts": opt.allow_shorts,
        "max_gross_per_ticker": opt.max_gross_per_ticker,
        "min_gross_divisor": 5.0,
        "position_sides": None,
        "manual_weights": None,
        "anchor_weights": None,
        "risk_free": app.features.risk_free_rate,
    }


def _maybe_apply_tilt(app, tickers: list[str], w: np.ndarray, kw: dict) -> tuple[np.ndarray, bool]:
    opt = app.risk.optimization
    weighting = opt.weighting
    if app.run is None or weighting not in ("optimised", "partial"):
        return w, False
    port = app.run.portfolio
    if not port.sentiment_magnitude_tilt:
        return w, False
    scores = finbert_scores_by_ticker(app, tickers, window_days=port.sentiment_sides_window_days)
    w2 = apply_sentiment_magnitude_tilt(
        w,
        tickers,
        scores,
        beta=port.sentiment_tilt_beta,
        cap=port.sentiment_tilt_cap,
        allow_shorts=kw["allow_shorts"],
        max_gross_per_ticker=kw["max_gross_per_ticker"],
        min_gross_divisor=kw["min_gross_divisor"],
        position_sides=kw["position_sides"],
    )
    return w2, True


def _max_sharpe_weights(
    app,
    tickers: list[str],
    mean_r: np.ndarray,
    cov: np.ndarray,
    cons: PortfolioConstraints,
) -> tuple[np.ndarray, bool, bool]:
    opt = app.risk.optimization
    weighting = opt.weighting
    kw = _run_portfolio_kw(app, tickers)

    if weighting == "optimised":
        w, ok = optimize_max_sharpe_gross(
            mean_r,
            cov,
            risk_free=kw["risk_free"],
            allow_shorts=kw["allow_shorts"],
            max_gross_per_ticker=kw["max_gross_per_ticker"],
            min_gross_divisor=kw["min_gross_divisor"],
        )
        w, tilted = _maybe_apply_tilt(app, tickers, w, kw)
        return w, ok, tilted

    if weighting == "partial":
        w, ok = optimize_partial_weights(
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
        w, tilted = _maybe_apply_tilt(app, tickers, w, kw)
        return w, ok, tilted

    if weighting in ("equal", "manual"):
        w = resolve_weights(weighting, tickers, **kw)
        return w, True, False

    w, converged = max_sharpe_weights(mean_r, cov, cons, risk_free=app.features.risk_free_rate)
    return w, converged, False


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_OPTIMIZATION_DIR)

    holdings_tickers = list(load_weights(app).keys())
    port = build_portfolio_returns(app)
    wide = load_returns_wide(holdings_tickers)
    tickers = align_optimization_tickers(wide, holdings_tickers, context="optimize_portfolios")
    excluded = [t for t in holdings_tickers if t not in tickers]

    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)

    mean_r, mu_meta = build_expected_returns(app, tickers)
    opt_cfg = app.risk.optimization
    cons = PortfolioConstraints(
        long_only=opt_cfg.long_only,
        min_weight=opt_cfg.min_weight,
        max_weight=opt_cfg.max_weight,
        leverage_cap=opt_cfg.leverage_cap,
    )

    w_min = min_variance_weights(cov, cons, len(tickers))
    w_sharpe, converged, tilt_applied = _max_sharpe_weights(app, tickers, mean_r, cov, cons)

    weight_rows = []
    stat_rows = []
    for label, w in [("min_variance", w_min), ("max_sharpe", w_sharpe)]:
        r, v, s = portfolio_stats(w, mean_r, cov)
        for t, wi in zip(tickers, w, strict=True):
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
        "excluded_tickers": excluded,
        "weighting": opt_cfg.weighting,
        "tilt_applied": tilt_applied,
        **mu_meta,
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
