"""Blended expected returns for portfolio optimization."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.portfolio.regime_policy import latest_hmm_regime_label, regime_sentiment_multiplier
from src.portfolio.sentiment_sides import finbert_scores_by_ticker
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import RISK_DATASET_PATH

LOGGER = get_logger(__name__)


def _agg_scalar(row: pl.DataFrame, column: str, *, default: float = 0.0) -> float:
    """Read one aggregated value from a single-row filter result."""
    if row.is_empty():
        return default
    val = row[column][0]
    if val is None:
        return default
    return float(val)


def _historical_mu(app: AppConfig, tickers: list[str]) -> np.ndarray:
    df = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .group_by("ticker")
        .agg(pl.col("returns").mean().alias("mu"))
        .collect()
    )
    mus: list[float] = []
    for t in tickers:
        row = df.filter(pl.col("ticker") == t)
        mu = _agg_scalar(row, "mu")
        if row.is_empty() or row["mu"][0] is None:
            LOGGER.warning(
                "Historical mu missing for %s in risk_dataset; using 0.0 for optimization",
                t,
            )
        mus.append(mu)
    return np.array(mus)


def _legacy_bullish_adjustment(tickers: list[str], app: AppConfig) -> np.ndarray:
    df = (
        pl.scan_parquet(RISK_DATASET_PATH)
        .filter(pl.col("ticker").is_in(tickers))
        .group_by("ticker")
        .agg(pl.col("bullish_ratio").mean().alias("sent"))
        .collect()
    )
    scale = app.risk.portfolio.sentiment_return_scale
    adj = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        row = df.filter(pl.col("ticker") == t)
        if row.height and row["sent"][0] is not None:
            adj[i] = scale * (float(row["sent"][0]) - 0.5)
    return adj


def _sentiment_mu_vector(
    tickers: list[str],
    scores: dict[str, float],
    hist_mu: np.ndarray,
    *,
    mode: str,
    scale: float,
) -> np.ndarray:
    sent = np.array([scores.get(t, 0.0) for t in tickers], dtype=float)
    if mode == "fixed":
        return scale * sent
    vols = np.abs(hist_mu)
    vols = np.where(vols < 1e-12, np.median(vols[vols > 1e-12]) if np.any(vols > 1e-12) else 1e-4, vols)
    return scale * sent * vols


def build_expected_returns(
    app: AppConfig,
    tickers: list[str],
    *,
    for_min_variance: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build mean return vector and metadata for optimization."""
    meta: dict[str, Any] = {}
    mu_hist = _historical_mu(app, tickers)
    port = app.run.portfolio if app.run is not None else None

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

    if port.use_legacy_bullish_mu and app.risk.optimization.use_sentiment_adjustment:
        mu = mu_hist + _legacy_bullish_adjustment(tickers, app)
        meta["mu_mode"] = "legacy_bullish"
        return mu, meta

    alpha = float(port.sentiment_mu_blend)
    if alpha <= 0:
        meta["mu_mode"] = "historical_only"
        meta["alpha_eff"] = 0.0
        return mu_hist, meta

    scores = finbert_scores_by_ticker(app, tickers, window_days=port.sentiment_mu_window_days)
    mu_sent = _sentiment_mu_vector(
        tickers,
        scores,
        mu_hist,
        mode=port.sentiment_mu_mode,
        scale=port.sentiment_mu_scale,
    )
    regime = latest_hmm_regime_label(app)
    regime_mult = regime_sentiment_multiplier(regime, port.regime_sentiment_mix)
    alpha_eff = alpha * regime_mult
    mu = (1.0 - alpha_eff) * mu_hist + alpha_eff * mu_sent
    meta.update(
        {
            "mu_mode": port.sentiment_mu_mode,
            "regime": regime,
            "alpha_eff": alpha_eff,
            "regime_mult": regime_mult,
        }
    )
    return mu, meta
