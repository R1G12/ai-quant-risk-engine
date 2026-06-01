"""Post-optimization magnitude tilt from FinBERT z-scores."""

from __future__ import annotations

import numpy as np

from src.portfolio.weights import (
    DEFAULT_MIN_GROSS_DIVISOR,
    apply_position_sides,
    enforce_min_gross_per_ticker,
    min_gross_per_ticker,
    normalize_gross_weights,
)


def apply_sentiment_magnitude_tilt(
    weights: np.ndarray,
    tickers: list[str],
    scores: dict[str, float],
    *,
    beta: float = 0.2,
    cap: float = 2.0,
    allow_shorts: bool = True,
    max_gross_per_ticker: float = 0.5,
    min_gross_divisor: float = DEFAULT_MIN_GROSS_DIVISOR,
    position_sides: dict[str, str] | None = None,
) -> np.ndarray:
    """Tilt weight magnitudes by z-scored sentiment; preserve gross budget and min floor."""
    w = np.asarray(weights, dtype=float).copy()
    if not scores:
        return w
    vals = np.array([scores.get(t, 0.0) for t in tickers], dtype=float)
    std = float(np.std(vals))
    if std < 1e-12:
        return w
    z = (vals - float(np.mean(vals))) / std
    z = np.clip(z, -cap, cap)
    tilted = w * (1.0 + beta * z)
    tilted = apply_position_sides(tilted, tickers, position_sides)
    min_w = min_gross_per_ticker(len(tickers), divisor=min_gross_divisor)
    tilted = enforce_min_gross_per_ticker(tilted, min_w)
    return normalize_gross_weights(
        tilted,
        allow_shorts=allow_shorts,
        max_gross_per_ticker=max_gross_per_ticker,
    )
