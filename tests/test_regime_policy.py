"""Tests for HMM regime sentiment multipliers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.portfolio.regime_policy import (
    latest_hmm_regime_label,
    regime_sentiment_multiplier,
)
from src.utils.config import load_app_config


def test_regime_sentiment_multiplier_lookup() -> None:
    mix = {"low": 0.4, "mid": 0.75, "high": 1.0}
    assert regime_sentiment_multiplier("low", mix) == pytest.approx(0.4)
    assert regime_sentiment_multiplier("high", mix) == pytest.approx(1.0)
    assert regime_sentiment_multiplier(None, mix) == pytest.approx(0.75)


def test_latest_hmm_regime_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "regimes.parquet"
    pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "regime": [0, 2],
            "regime_label": ["mid", "high"],
        }
    ).write_parquet(path)
    monkeypatch.setattr("src.portfolio.regime_policy.RISK_REGIMES_PATH", path)
    app = load_app_config()
    assert latest_hmm_regime_label(app) == "high"
