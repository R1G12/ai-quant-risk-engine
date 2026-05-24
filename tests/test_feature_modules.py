"""Unit tests for feature engineering modules."""

import polars as pl

from src.features.liquidity import add_liquidity_features
from src.features.returns import add_returns
from src.features.risk_ratios import add_rolling_sharpe
from src.features.volatility import add_volatility_features
from src.utils.config import FeatureConfig


def _ohlcv_frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 2, 1),
                interval="1d",
                eager=True,
            ),
            "ticker": ["AAPL"] * 32,
            "close": [100.0 + i * 0.5 for i in range(32)],
            "volume": [1_000_000.0 + i * 1000 for i in range(32)],
        }
    ).lazy()


def test_add_returns_columns() -> None:
    out = _ohlcv_frame().pipe(add_returns).collect()
    assert "returns" in out.columns
    assert "log_returns" in out.columns


def test_volatility_and_sharpe_and_liquidity() -> None:
    cfg = FeatureConfig(
        volatility_window=5,
        sharpe_window=5,
        risk_free_rate=0.02,
        annualization_factor=252,
        momentum_window=10,
        sma_windows=[10, 20],
        correlation_window=10,
    )
    lf = (
        _ohlcv_frame()
        .pipe(add_returns)
        .pipe(add_volatility_features, cfg)
        .pipe(add_rolling_sharpe, cfg)
        .pipe(add_liquidity_features, cfg.volatility_window)
    )
    out = lf.collect()
    assert "volatility" in out.columns
    assert "rolling_sharpe" in out.columns
    assert "volume_zscore" in out.columns
