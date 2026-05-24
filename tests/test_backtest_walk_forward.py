"""Walk-forward backtest tests."""

import polars as pl

from src.backtesting.engine.walk_forward import run_historical_backtest
from src.backtesting.rebalancing.calendar import rebalance_mask, walk_forward_window_ids
from src.utils.config import load_app_config


def _sample_wide(n: int = 80) -> pl.DataFrame:
    ts = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 4, 30), interval="1d", eager=True).cast(
        pl.Datetime
    )
    ts = ts.head(n)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "AAPL": [0.001] * len(ts),
            "MSFT": [0.0005] * len(ts),
        }
    )


def test_costs_only_on_rebalance_days() -> None:
    app = load_app_config()
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    eq, _, _ = run_historical_backtest(_sample_wide(), weights, app)
    non_rebal = eq.filter(~pl.col("is_rebalance"))
    assert non_rebal.height > 0
    assert non_rebal["cost"].max() == 0.0
    rebal = eq.filter(pl.col("is_rebalance"))
    assert rebal.height > 0
    assert rebal["cost"].min() > 0.0


def test_unique_timestamps_in_equity() -> None:
    app = load_app_config()
    eq, _, _ = run_historical_backtest(_sample_wide(), {"AAPL": 0.5, "MSFT": 0.5}, app)
    dupes = eq.group_by("timestamp").len().filter(pl.col("len") > 1)
    assert dupes.height == 0


def test_walk_forward_multiple_windows() -> None:
    app = load_app_config()
    eq, _, _ = run_historical_backtest(_sample_wide(80), {"AAPL": 1.0}, app)
    test_windows = eq.filter(pl.col("window_id") >= 0)["window_id"].unique().to_list()
    assert len(test_windows) >= 2


def test_rebalance_mask_monthly() -> None:
    ts = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 3, 1), interval="1d", eager=True).cast(
        pl.Datetime
    )
    mask = rebalance_mask(ts, "monthly")
    assert mask.sum() >= 3


def test_window_ids_warmup() -> None:
    ids = walk_forward_window_ids(50, train_days=10, test_days=5)
    assert ids[:10] == [-1] * 10
    assert ids[10] == 0
