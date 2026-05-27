"""Chart registry and validation tests."""

import polars as pl

from src.analytics.charts.context import ChartContext
from src.analytics.charts.registry import build_chart_registry
from src.analytics.validation import assert_equity_sane, assert_unique_timestamps
from src.utils.config import load_app_config


def test_chart_registry_builds_without_error() -> None:
    app = load_app_config()
    registry = build_chart_registry()
    assert len(registry) >= 10
    built = 0
    ctx = ChartContext(app=app)
    for spec in registry:
        fig = spec.builder(ctx)
        if fig is not None:
            built += 1
    assert built >= 1


def test_assert_unique_timestamps_raises() -> None:
    df = pl.DataFrame({"timestamp": [1, 1, 2]})
    try:
        assert_unique_timestamps(df, "timestamp")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_assert_equity_sane_passes() -> None:
    rets = [0.001] * 10
    equity = []
    e = 1.0
    for r in rets:
        e *= 1 + r
        equity.append(e)
    df = pl.DataFrame(
        {
            "timestamp": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), interval="1d", eager=True),
            "portfolio_return": rets,
            "equity": equity,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime))
    assert_equity_sane(df)
