"""Feature engineering unit tests."""

import polars as pl

from src.features.returns import add_returns
from src.utils.config import FeatureConfig
from src.features.volatility import add_volatility_features


def test_add_returns_vectorized() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "ticker": ["AAPL"] * 5,
            "open": [100.0, 101.0, 102.0, 101.0, 103.0],
            "high": [101.0, 102.0, 103.0, 102.0, 104.0],
            "low": [99.0, 100.0, 101.0, 100.0, 102.0],
            "close": [100.0, 101.0, 102.0, 101.0, 103.0],
            "volume": [1e6] * 5,
        }
    ).lazy()

    out = add_returns(lf).collect()
    assert "returns" in out.columns
    assert "log_returns" in out.columns
    assert out["returns"].null_count() >= 1


def test_volatility_features() -> None:
    cfg = FeatureConfig(volatility_window=2)
    lf = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 4),
                interval="1d",
                eager=True,
            ),
            "ticker": ["AAPL"] * 4,
            "returns": [None, 0.01, -0.02, 0.03],
        }
    ).lazy()
    out = add_volatility_features(lf, cfg).collect()
    assert "volatility" in out.columns
