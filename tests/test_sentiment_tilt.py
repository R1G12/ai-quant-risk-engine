"""Tests for post-optimization sentiment tilt."""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.sentiment_tilt import apply_sentiment_magnitude_tilt


def test_tilt_preserves_gross_budget() -> None:
    tickers = ["A", "B", "C", "D"]
    w = np.array([0.25, 0.25, 0.25, 0.25])
    scores = {"A": 0.8, "B": 0.2, "C": -0.5, "D": -0.9}
    out = apply_sentiment_magnitude_tilt(
        w,
        tickers,
        scores,
        beta=0.3,
        max_gross_per_ticker=0.5,
        min_gross_divisor=5.0,
    )
    assert np.sum(np.abs(out)) == pytest.approx(1.0, rel=1e-4)
