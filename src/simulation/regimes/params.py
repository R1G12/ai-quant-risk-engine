"""Map Phase 3 regimes to simulation parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.utils.config import AppConfig
from src.utils.paths import RISK_PORTFOLIO_RETURNS_PATH, RISK_REGIMES_PATH


def load_regime_params(app: AppConfig) -> dict[str, dict[str, float]]:
    """Per-regime mu and sigma from historical portfolio returns."""
    params: dict[str, dict[str, float]] = {}
    if not RISK_REGIMES_PATH.is_file() or not RISK_PORTFOLIO_RETURNS_PATH.is_file():
        return {"default": {"mu": 0.0, "sigma": 0.02}}

    regimes = pl.read_parquet(RISK_REGIMES_PATH)
    returns = pl.read_parquet(RISK_PORTFOLIO_RETURNS_PATH)
    df = regimes.join(returns, on="timestamp", how="inner")
    ann = app.features.annualization_factor

    for label in df["regime_label"].unique().to_list():
        sub = df.filter(pl.col("regime_label") == label)["portfolio_return"].drop_nulls()
        if sub.len() == 0:
            continue
        std = sub.std()
        sigma = float(std) * np.sqrt(ann) if std is not None and std == std else 0.15
        params[str(label)] = {
            "mu": float(sub.mean()) * ann,
            "sigma": sigma,
        }
    return params or {"default": {"mu": 0.0, "sigma": 0.02}}
