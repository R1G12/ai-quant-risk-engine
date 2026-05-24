"""Portfolio wealth path helpers."""

from __future__ import annotations

import numpy as np
import polars as pl


def weights_from_dict(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    """Ordered weight vector."""
    return np.array([weights.get(t, 0.0) for t in tickers])


def portfolio_paths_from_asset_paths(
    asset_paths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Combine (n_paths, n_assets, T) -> (n_paths, T)."""
    return np.einsum("pat,a->pt", asset_paths, weights)
