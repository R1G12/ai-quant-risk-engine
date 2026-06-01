"""Tests for Signals HMM regime chart helpers."""

from __future__ import annotations

import polars as pl

from src.dashboards.core.signals_loaders import (
    HMM_REGIME_LABELS,
    REGIME_HISTORY_OBS,
    regime_history_window,
    regimes_missing_in_window,
)


def test_regime_history_window_tail() -> None:
    df = pl.DataFrame(
        {
            "timestamp": list(range(400)),
            "regime_label": ["mid"] * 400,
        }
    )
    win = regime_history_window(df, n_obs=REGIME_HISTORY_OBS)
    assert win.height == REGIME_HISTORY_OBS
    assert win["timestamp"][0] == 35


def test_regimes_missing_in_window() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "regime_label": ["mid", "mid", "high"],
        }
    )
    assert regimes_missing_in_window(df) == ["low"]
    assert regimes_missing_in_window(df.filter(pl.col("regime_label") == "mid")) == [
        "low",
        "high",
    ]
    assert regimes_missing_in_window(
        pl.DataFrame({"regime_label": list(HMM_REGIME_LABELS)})
    ) == []
