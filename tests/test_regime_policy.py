"""Tests for HMM regime sentiment multiplier."""

from __future__ import annotations

from pathlib import Path

import pytest
import polars as pl

from src.portfolio.regime_policy import (
    latest_hmm_regime_label,
    regime_sentiment_multiplier,
)
from src.utils.config import load_app_config


def test_regime_multiplier_defaults() -> None:
    assert regime_sentiment_multiplier("low") == pytest.approx(0.4)
    assert regime_sentiment_multiplier("high") == pytest.approx(1.0)
    assert regime_sentiment_multiplier(None) == 1.0


def test_latest_regime_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "regimes.parquet"
    pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "regime": [0, 1, 2],
            "regime_label": ["low", "mid", "high"],
        }
    ).write_parquet(p)
    monkeypatch.setattr("src.portfolio.regime_policy.RISK_REGIMES_PATH", p)
    app = load_app_config()
    assert latest_hmm_regime_label(app) == "high"
