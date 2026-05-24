"""Merge features tests."""

import polars as pl

from src.features.merge import merge_feature_frames


def test_merge_feature_frames_join_keys() -> None:
    ts = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 1, 3),
        interval="1d",
        eager=True,
    )
    returns = pl.DataFrame(
        {
            "timestamp": ts,
            "ticker": ["AAPL"] * 3,
            "close": [100.0, 101.0, 102.0],
            "returns": [None, 0.01, 0.0099],
            "log_returns": [None, 0.00995, 0.00985],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "volume": [1e6, 1e6, 1e6],
        }
    ).lazy()
    vol = pl.DataFrame(
        {
            "timestamp": ts,
            "ticker": ["AAPL"] * 3,
            "volatility": [None, 0.01, 0.02],
            "annualized_volatility": [None, 0.15, 0.30],
            "rolling_sharpe": [None, 1.0, 1.1],
        }
    ).lazy()
    tech = pl.DataFrame(
        {
            "timestamp": ts,
            "ticker": ["AAPL"] * 3,
            "sma_10": [100.0, 100.5, 101.0],
            "momentum": [0.0, 0.01, 0.02],
        }
    ).lazy()
    sent = pl.DataFrame(
        {
            "timestamp": ts,
            "ticker": ["AAPL"] * 3,
            "confidence": [0.9, 0.8, 0.85],
            "bullish_ratio": [0.6, 0.5, 0.7],
            "article_count": [1, 2, 1],
        }
    ).lazy()

    merged = merge_feature_frames(returns, vol, tech, sent).collect()
    assert merged.height == 3
    assert "volatility" in merged.columns
    assert "bullish_ratio" in merged.columns
