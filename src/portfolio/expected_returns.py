"""Blended expected returns for portfolio optimization (historical + FinBERT)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.portfolio.regime_policy import latest_hmm_regime_label, regime_sentiment_multiplier
from src.portfolio.sentiment_sides import ticker_sentiment_scores
from src.utils.config import AppConfig, PortfolioRunConfig
from src.utils.paths import RISK_DATASET_PATH


def _portfolio_config(app: AppConfig) -> PortfolioRunConfig | None:
    return app.run.portfolio if app.run is not None else None


def historical_mean_returns(
    tickers: list[str],
    app: AppConfig,
    *,
    window_days: int | None = None,
) -> np.ndarray:
    """Mean daily returns per ticker from the merged risk dataset."""
    lf = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .select(["ticker", "timestamp", "returns"])
    )
    if window_days is not None and window_days > 0:
        max_ts = lf.select(pl.col("timestamp").max()).collect().item()
        if max_ts is not None:
            cutoff = max_ts - pl.duration(days=int(window_days))
            lf = lf.filter(pl.col("timestamp") >= cutoff)

    df = lf.group_by("ticker").agg(pl.col("returns").mean().alias("mu")).collect()
    out: list[float] = []
    for t in tickers:
        row = df.filter(pl.col("ticker") == t)
        out.append(float(row["mu"][0]) if row.height else 0.0)
    return np.asarray(out, dtype=float)


def recent_volatility(
    tickers: list[str],
    app: AppConfig,
    *,
    window: int = 63,
) -> np.ndarray:
    """Per-ticker daily return std over the last ``window`` observations."""
    lf = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .select(["ticker", "timestamp", "returns"])
    )
    max_ts = lf.select(pl.col("timestamp").max()).collect().item()
    if max_ts is not None and window > 0:
        cutoff = max_ts - pl.duration(days=int(window))
        lf = lf.filter(pl.col("timestamp") >= cutoff)

    df = lf.group_by("ticker").agg(pl.col("returns").std().alias("sigma")).collect()
    sigmas: list[float] = []
    for t in tickers:
        row = df.filter(pl.col("ticker") == t)
        s = float(row["sigma"][0]) if row.height and row["sigma"][0] is not None else 0.0
        sigmas.append(s if s > 1e-12 else 1e-4)
    return np.asarray(sigmas, dtype=float)


def sentiment_mean_returns(
    tickers: list[str],
    app: AppConfig,
    *,
    window_days: int,
    mode: str,
    scale: float,
    vol: np.ndarray | None = None,
) -> np.ndarray:
    """Map FinBERT scores to daily expected-return units."""
    scores = ticker_sentiment_scores(app, tickers, window_days=window_days)
    mus: list[float] = []
    for i, t in enumerate(tickers):
        s = scores.get(t)
        if s is None:
            mus.append(0.0)
            continue
        if mode == "vol_scaled":
            sigma = float(vol[i]) if vol is not None else 1e-4
            mus.append(scale * s * sigma)
        else:
            mus.append(scale * s * 1e-4)
    return np.asarray(mus, dtype=float)


def _legacy_bullish_adjustment(tickers: list[str], app: AppConfig) -> np.ndarray:
    """Full-sample bullish_ratio nudge (pre-blend behavior)."""
    df = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .group_by("ticker")
        .agg(pl.col("bullish_ratio").mean().alias("sent"))
        .collect()
    )
    scale = app.risk.portfolio.sentiment_return_scale
    adj = []
    for t in tickers:
        row = df.filter(pl.col("ticker") == t)
        if row.height == 0 or row["sent"][0] is None:
            adj.append(0.0)
        else:
            adj.append(scale * (float(row["sent"][0]) - 0.5))
    return np.asarray(adj, dtype=float)


def build_expected_returns(
    tickers: list[str],
    app: AppConfig,
    *,
    for_min_variance: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build μ vector with optional sentiment blend and HMM regime scaling.

    When ``for_min_variance`` is True, returns historical μ only (no sentiment).
    """
    port = _portfolio_config(app)
    meta: dict[str, Any] = {
        "for_min_variance": for_min_variance,
        "regime": None,
        "regime_mult": 1.0,
        "alpha": 0.0,
        "alpha_eff": 0.0,
        "beta_eff": 1.0,
        "mu_mode": "historical_only",
    }

    hist_window = port.sentiment_mu_hist_window_days if port else None
    mu_hist = historical_mean_returns(tickers, app, window_days=hist_window)

    if for_min_variance:
        meta["mu_mode"] = "historical_only"
        return mu_hist, meta

    if port is None:
        if app.risk.optimization.use_sentiment_adjustment:
            mu = mu_hist + _legacy_bullish_adjustment(tickers, app)
            meta["mu_mode"] = "legacy_bullish"
            return mu, meta
        meta["mu_mode"] = "historical_only"
        return mu_hist, meta

    regime = latest_hmm_regime_label(app)
    regime_mult = regime_sentiment_multiplier(regime, port.regime_sentiment_mix)
    meta["regime"] = regime
    meta["regime_mult"] = regime_mult

    alpha = float(port.sentiment_mu_blend)
    alpha_eff = alpha * regime_mult
    meta["alpha"] = alpha
    meta["alpha_eff"] = alpha_eff
    meta["beta_eff"] = regime_mult

    if alpha_eff <= 1e-12 and not port.use_legacy_bullish_mu:
        meta["mu_mode"] = "historical_only"
        return mu_hist, meta

    if alpha_eff <= 1e-12 and port.use_legacy_bullish_mu:
        mu = mu_hist + _legacy_bullish_adjustment(tickers, app)
        meta["mu_mode"] = "legacy_bullish"
        return mu, meta

    vol = recent_volatility(tickers, app, window=app.risk.portfolio.rolling_metrics_window)
    mu_sent = sentiment_mean_returns(
        tickers,
        app,
        window_days=port.sentiment_mu_window_days,
        mode=port.sentiment_mu_mode,
        scale=port.sentiment_mu_scale,
        vol=vol,
    )
    mu = (1.0 - alpha_eff) * mu_hist + alpha_eff * mu_sent
    meta["mu_mode"] = port.sentiment_mu_mode
    return mu, meta
