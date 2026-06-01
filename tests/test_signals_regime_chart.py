"""Tests for Signals HMM regime chart helpers."""

from __future__ import annotations

import polars as pl

from src.dashboards.core.signals_loaders import (
    HMM_REGIME_LABELS,
    REGIME_HISTORY_OBS,
    regime_day_counts,
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


def test_regimes_missing_in_window() -> None:
    df = pl.DataFrame({"regime_label": ["mid", "mid", "high"]})
    missing = regimes_missing_in_window(df)
    assert "low" in missing
    assert "mid" not in missing


def test_regime_day_counts_all_labels() -> None:
    df = pl.DataFrame({"regime_label": ["mid", "mid", "high"]})
    counts = regime_day_counts(df)
    assert counts.height == len(HMM_REGIME_LABELS)
    assert set(counts["regime_label"].to_list()) == set(HMM_REGIME_LABELS)
