"""Tests for post-optimization sentiment magnitude tilt."""

from __future__ import annotations

import numpy as np

from src.portfolio.sentiment_tilt import apply_sentiment_magnitude_tilt


def test_tilt_increases_bullish_weight_share() -> None:
    tickers = ["BULL", "BEAR"]
    w = np.array([0.5, -0.5])
    scores = {"BULL": 0.8, "BEAR": -0.8}
    w2 = apply_sentiment_magnitude_tilt(
        w,
        tickers,
        scores,
        beta=0.5,
        cap=3.0,
        beta_eff=1.0,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        min_gross_divisor=5.0,
    )
    assert abs(np.sum(np.abs(w2)) - 1.0) < 1e-5
    assert abs(w2[0]) > abs(w2[1])


def test_tilt_respects_cap() -> None:
    tickers = ["A", "B", "C"]
    w = np.ones(3) / 3
    scores = {"A": 1.0, "B": 0.0, "C": -1.0}
    w2 = apply_sentiment_magnitude_tilt(
        w,
        tickers,
        scores,
        beta=10.0,
        cap=1.5,
        beta_eff=1.0,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        min_gross_divisor=5.0,
    )
    ratios = np.abs(w2) / (np.abs(w) + 1e-12)
    assert float(ratios.max()) <= 1.5 + 1e-6
