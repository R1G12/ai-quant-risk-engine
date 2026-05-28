"""Covariance matrix generation with optional shrinkage."""

from __future__ import annotations

import numpy as np
import polars as pl


def tickers_in_wide(returns_wide: pl.DataFrame, tickers: list[str]) -> list[str]:
    """Return tickers that appear as return columns in a wide frame."""
    return [c for c in tickers if c in returns_wide.columns]


def sample_covariance_matrix(returns_wide: pl.DataFrame, tickers: list[str]) -> np.ndarray:
    """Sample covariance from wide return matrix (columns = tickers)."""
    cols = tickers_in_wide(returns_wide, tickers)
    if len(cols) < 2:
        missing = [t for t in tickers if t not in cols]
        raise ValueError(
            f"Need at least 2 tickers with return history for covariance; "
            f"found {len(cols)} ({cols}). "
            f"Missing or empty: {missing[:10]}{'...' if len(missing) > 10 else ''}. "
            "Re-run market ingest after `aqre prepare` (check MARKET_SOURCE and configs/run.yaml tickers)."
        )
    arr = returns_wide.select(cols).to_numpy()
    arr = np.nan_to_num(arr, nan=0.0)
    cov = np.cov(arr, rowvar=False)
    return np.atleast_2d(np.asarray(cov, dtype=float))


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
