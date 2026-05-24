"""Volatility engine tests."""

import polars as pl

from src.risk.volatility.ewma import add_ewma_volatility
from src.risk.volatility.regimes_simple import add_vol_regime_flags
from src.utils.config import VolatilityRiskConfig


def test_ewma_vol_column() -> None:
    cfg = VolatilityRiskConfig(ewma_span=5, rolling_vol_window=5, long_vol_window=10)
    lf = pl.DataFrame({"timestamp": range(20), "portfolio_return": [0.01] * 20}).lazy()
    out = add_ewma_volatility(lf, cfg).collect()
    assert "ewma_vol" in out.columns
    assert out["ewma_vol"].drop_nulls().len() > 0


def test_vol_regime_flag() -> None:
    cfg = VolatilityRiskConfig(regime_zscore_threshold=0.5, long_vol_window=5)
    lf = pl.DataFrame(
        {
            "timestamp": range(30),
            "portfolio_return": [0.01] * 15 + [-0.02] * 15,
            "ewma_vol": [0.01] * 15 + [0.05] * 15,
        }
    ).lazy()
    out = add_vol_regime_flags(lf, cfg, vol_col="ewma_vol").collect()
    assert "vol_regime_flag" in out.columns
