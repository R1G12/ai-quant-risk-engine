"""DVC stage: optimize_portfolios."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl

from src.features.wide_returns import load_returns_wide
from src.portfolio.expected_returns import build_expected_returns
from src.portfolio.regime_policy import latest_hmm_regime_label, regime_sentiment_multiplier
from src.portfolio.sentiment_sides import effective_position_sides, ticker_sentiment_scores
from src.portfolio.sentiment_tilt import apply_sentiment_magnitude_tilt
from src.portfolio.weights import (
    optimize_max_sharpe_gross,
    optimize_partial_weights,
    resolve_weights,
)
from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix
from src.risk.optimization.constraints import PortfolioConstraints
from src.risk.optimization.markowitz import max_sharpe_weights, min_variance_weights, portfolio_stats
from src.risk.pipeline._io import write_single_parquet
from src.risk.portfolio.holdings import read_holdings_weights
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


def _mean_returns_with_sentiment(app, tickers: list[str]) -> np.ndarray:
    """Backward-compatible helper: blended μ for max-Sharpe / frontier."""
    mu, _ = build_expected_returns(tickers, app)
    return mu


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


def _maybe_apply_sentiment_tilt(
    app,
    tickers: list[str],
    w: np.ndarray,
    kw: dict,
    mu_meta: dict[str, Any],
) -> np.ndarray:
    port = app.run.portfolio if app.run is not None else None
    if port is None or not port.sentiment_magnitude_tilt:
        return w

    scores = ticker_sentiment_scores(app, tickers, window_days=port.sentiment_mu_window_days)
    if not scores:
        return w

    beta_eff = float(mu_meta.get("beta_eff", 1.0))
    w_tilt = apply_sentiment_magnitude_tilt(
        w,
        tickers,
        scores,
        beta=port.sentiment_tilt_beta,
        cap=port.sentiment_tilt_cap,
        beta_eff=beta_eff,
        allow_shorts=kw["allow_shorts"],
        max_gross_per_ticker=kw["max_gross_per_ticker"],
        min_gross_divisor=kw["min_gross_divisor"],
        position_sides=kw.get("position_sides"),
    )
    LOGGER.info(
        "Applied sentiment magnitude tilt (beta=%.3f, beta_eff=%.3f)",
        port.sentiment_tilt_beta,
        beta_eff,
    )
    return w_tilt


def _max_sharpe_weights(
    app,
    tickers: list[str],
    mean_r: np.ndarray,
    cov: np.ndarray,
    cons: PortfolioConstraints,
    mu_meta: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    opt = app.risk.optimization
    weighting = opt.weighting
    kw = _run_portfolio_kw(app)
    kw["position_sides"] = effective_position_sides(app, tickers, kw.get("position_sides"))

    if weighting == "optimised":
        w, ok = optimize_max_sharpe_gross(
            mean_r,
            cov,
            tickers,
            risk_free=kw["risk_free"],
            allow_shorts=kw["allow_shorts"],
            max_gross_per_ticker=kw["max_gross_per_ticker"],
            min_gross_divisor=kw["min_gross_divisor"],
            position_sides=kw["position_sides"],
        )
        w = _maybe_apply_sentiment_tilt(app, tickers, w, kw, mu_meta)
        return w, ok

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
        w = _maybe_apply_sentiment_tilt(app, tickers, w, kw, mu_meta)
        return w, ok

    if weighting in ("equal", "manual"):
        w = resolve_weights(weighting, tickers, **kw)
        return w, True

    w, converged = max_sharpe_weights(mean_r, cov, cons, risk_free=app.features.risk_free_rate)
    return w, converged


def run() -> None:
    app = load_app_config()
    ensure_dir(RISK_OPTIMIZATION_DIR)

    tickers = list(read_holdings_weights(app).keys())
    port_returns = build_portfolio_returns(app)
    wide = load_returns_wide(tickers)

    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)

    mean_r, mu_meta = build_expected_returns(tickers, app)
    mean_r_hist, _ = build_expected_returns(tickers, app, for_min_variance=True)

    regime = latest_hmm_regime_label(app)
    if app.run is not None:
        regime_mult = regime_sentiment_multiplier(regime, app.run.portfolio.regime_sentiment_mix)
        LOGGER.info(
            "Expected returns: mode=%s alpha_eff=%.4f regime=%s regime_mult=%.3f",
            mu_meta.get("mu_mode"),
            float(mu_meta.get("alpha_eff", 0.0)),
            regime,
            regime_mult,
        )

    opt_cfg = app.risk.optimization
    cons = PortfolioConstraints(
        long_only=opt_cfg.long_only,
        min_weight=opt_cfg.min_weight,
        max_weight=opt_cfg.max_weight,
        leverage_cap=opt_cfg.leverage_cap,
    )

    w_min = min_variance_weights(cov, cons, len(tickers))
    w_sharpe, converged = _max_sharpe_weights(app, tickers, mean_r, cov, cons, mu_meta)

    weight_rows = []
    stat_rows = []
    for label, w, mu_vec in [
        ("min_variance", w_min, mean_r_hist),
        ("max_sharpe", w_sharpe, mean_r),
    ]:
        r, v, s = portfolio_stats(w, mu_vec, cov)
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

    tilt_applied = bool(
        app.run is not None
        and app.run.portfolio.sentiment_magnitude_tilt
        and opt_cfg.weighting in ("optimised", "partial")
    )
    meta: dict[str, Any] = {
        "converged": converged,
        "shrinkage": opt_cfg.shrinkage,
        "tickers": tickers,
        "weighting": opt_cfg.weighting,
        "regime": mu_meta.get("regime"),
        "regime_mult": mu_meta.get("regime_mult"),
        "alpha_eff": mu_meta.get("alpha_eff"),
        "beta_eff": mu_meta.get("beta_eff"),
        "sentiment_mu_blend": mu_meta.get("alpha"),
        "mu_mode": mu_meta.get("mu_mode"),
        "tilt_applied": tilt_applied,
    }
    RISK_OPT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RISK_OPT_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_stage_metrics(
        METRICS_DIR / "risk_optimization",
        {"converged": int(converged), "portfolio_vol": float(port_returns["portfolio_return"].std())},
    )
    LOGGER.info("Optimization complete", extra={"converged": converged})


if __name__ == "__main__":
    run()
