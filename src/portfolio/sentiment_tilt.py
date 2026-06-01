"""Post-optimization sentiment magnitude tilt on portfolio weights."""

from __future__ import annotations

import numpy as np

from src.portfolio.weights import enforce_min_gross_per_ticker, min_gross_per_ticker, normalize_gross_weights


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def apply_sentiment_magnitude_tilt(
    weights: np.ndarray,
    tickers: list[str],
    scores: dict[str, float],
    *,
    beta: float,
    cap: float,
    beta_eff: float = 1.0,
    allow_shorts: bool = True,
    max_gross_per_ticker: float = 0.5,
    min_gross_divisor: float = 5.0,
    position_sides: dict[str, str] | None = None,
) -> np.ndarray:
    """Tilt gross magnitudes toward bullish FinBERT scores, then re-normalize."""
    if beta <= 1e-12 or not tickers:
        return np.asarray(weights, dtype=float)

    z = _zscore([scores.get(t, 0.0) for t in tickers])
    beta_use = beta * beta_eff
    mult = np.clip(1.0 + beta_use * z, 1.0 / cap, cap)
    w = np.asarray(weights, dtype=float) * mult
    w = normalize_gross_weights(
        w,
        allow_shorts=allow_shorts,
        max_gross_per_ticker=max_gross_per_ticker,
    )
    min_w = min_gross_per_ticker(len(tickers), divisor=min_gross_divisor)
    return enforce_min_gross_per_ticker(
        w,
        tickers,
        min_w,
        allow_shorts=allow_shorts,
        max_gross_per_ticker=max_gross_per_ticker,
        position_sides=position_sides,
    )
