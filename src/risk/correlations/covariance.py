"""Covariance matrix generation with optional shrinkage."""

from __future__ import annotations

import numpy as np
import polars as pl


def sample_covariance_matrix(returns_wide: pl.DataFrame, tickers: list[str]) -> np.ndarray:
    """Sample covariance from wide return matrix (columns = tickers)."""
    cols = [c for c in tickers if c in returns_wide.columns]
    arr = returns_wide.select(cols).to_numpy()
    arr = np.nan_to_num(arr, nan=0.0)
    return np.cov(arr, rowvar=False)


def ledoit_wolf_shrinkage(cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward diagonal target (numpy/scipy boundary)."""
    n = cov.shape[0]
    mu = np.trace(cov) / n
    prior = mu * np.eye(n)
    delta = cov - prior
    shrink = 0.1
    return prior + (1.0 - shrink) * delta


def covariance_to_long(
    timestamp: object,
    tickers: list[str],
    cov: np.ndarray,
    metric: str = "cov",
) -> pl.DataFrame:
    """Tidy long-format covariance storage."""
    rows = []
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset_i": a,
                    "asset_j": b,
                    "value": float(cov[i, j]),
                    "metric": metric,
                }
            )
    return pl.DataFrame(rows)
