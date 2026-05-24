"""Holdings and portfolio return tests."""

import polars as pl

from src.risk.portfolio.exposures import exposure_table


def test_exposure_table_weights() -> None:
    w = {"AAPL": 0.6, "MSFT": 0.4}
    df = exposure_table(w)
    assert abs(df["weight"].sum() - 1.0) < 1e-6
