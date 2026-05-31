"""Calibrate simulation inputs from Phase 3 outputs."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.features.wide_returns import load_returns_wide
from src.risk.correlations.covariance import ledoit_wolf_shrinkage, sample_covariance_matrix
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils.paths import RISK_CORRELATIONS_DIR, RISK_PORTFOLIO_RETURNS_PATH


def calibrate_gbm(app: AppConfig) -> tuple[float, float]:
    """Historical mu, sigma from portfolio returns."""
    port = pl.read_parquet(RISK_PORTFOLIO_RETURNS_PATH)
    r = port["portfolio_return"].drop_nulls()
    ann = app.features.annualization_factor
    return float(r.mean()) * ann, float(r.std()) * np.sqrt(ann)


def calibrate_multivariate(app: AppConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mu vector and covariance from risk dataset."""
    tickers = list(load_weights(app).keys())
    wide = load_returns_wide(tickers)
    cov = sample_covariance_matrix(wide, tickers)
    if app.risk.optimization.shrinkage == "ledoit_wolf":
        cov = ledoit_wolf_shrinkage(cov)
    ann = app.features.annualization_factor
    mu = np.array([float(wide[t].mean()) * ann for t in tickers if t in wide.columns])
    return mu, cov, tickers
