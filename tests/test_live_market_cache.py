"""Tests for live market cache invalidation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.analytics.live_market_cache import (
    invalidate_live_cache,
    is_live_ready,
    reset_live_state_for_tests,
)
from src.utils.paths import MARKET_LIVE_RAW_PATH


def test_invalidate_live_cache_removes_active_file(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "market_live.parquet"
    monkeypatch.setattr("src.analytics.live_market_cache.MARKET_LIVE_RAW_PATH", cache)
    reset_live_state_for_tests()
    pl.DataFrame({"ticker": ["AAPL"], "close": [1.0]}).write_parquet(cache)

    assert cache.is_file()
    assert is_live_ready()

    invalidate_live_cache()
    assert not cache.is_file()
    assert cache.with_suffix(".parquet.bak").is_file()
    assert not is_live_ready()

    reset_live_state_for_tests()
