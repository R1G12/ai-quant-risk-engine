"""Tests for FinBERT-derived position sides."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.portfolio.sentiment_sides import (
    effective_position_sides,
    merge_position_sides_from_sentiment,
)
from src.portfolio.weights import optimize_max_sharpe_gross
from src.utils.config import PortfolioRunConfig, RunConfig, RunMarketOverrides, load_app_config


def test_merge_respects_explicit_over_sentiment() -> None:
    merged = merge_position_sides_from_sentiment(
        ["PDD", "USO"],
        {"PDD": "long"},
        {"PDD": -0.9, "USO": -0.1},
        bear_threshold=-0.3,
        bull_threshold=0.3,
    )
    assert merged["PDD"] == "long"
    assert "USO" not in merged


def test_merge_short_only_below_bear_threshold() -> None:
    merged = merge_position_sides_from_sentiment(
        ["PDD", "USO"],
        {},
        {"PDD": -0.9, "USO": -0.1},
        bear_threshold=-0.3,
        bull_threshold=0.3,
    )
    assert merged["PDD"] == "short"
    assert "USO" not in merged


def test_effective_position_sides_from_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent_path = tmp_path / "sentiment.parquet"
    end = date.today()
    rows = [
        {
            "date": (end - timedelta(days=1)).isoformat(),
            "source": "Reuters",
            "text": "bad",
            "ticker": "PDD",
            "sentiment_label": "negative",
            "sentiment_score": 0.95,
        },
        {
            "date": (end - timedelta(days=1)).isoformat(),
            "source": "Reuters",
            "text": "good",
            "ticker": "IWM",
            "sentiment_label": "positive",
            "sentiment_score": 0.95,
        },
    ]
    pl.DataFrame(rows).write_parquet(sent_path)
    monkeypatch.setattr("src.portfolio.sentiment_sides.PROCESSED_SENTIMENT_PATH", sent_path)

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        """
mode: demo
market:
  tickers: [PDD, IWM]
portfolio:
  weighting: optimised
  allow_shorts: true
  sentiment_position_sides: true
""".strip(),
        encoding="utf-8",
    )
    app = load_app_config(run_yaml)
    sides = effective_position_sides(app, ["PDD", "IWM"])
    assert sides is not None
    assert sides["PDD"] == "short"
    assert sides["IWM"] == "long"


def test_max_sharpe_applies_sentiment_short_side() -> None:
    n = 3
    tickers = ["A", "B", "C"]
    mu = np.array([0.01, 0.005, 0.001])
    cov = np.diag([0.0004, 0.0004, 0.0004])
    w, ok = optimize_max_sharpe_gross(
        mu,
        cov,
        tickers,
        allow_shorts=True,
        max_gross_per_ticker=0.5,
        position_sides={"C": "short"},
    )
    assert ok
    assert w[tickers.index("C")] < 0


def test_effective_skips_inference_when_allow_shorts_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent_path = tmp_path / "sentiment.parquet"
    end = date.today()
    pl.DataFrame(
        [
            {
                "date": (end - timedelta(days=1)).isoformat(),
                "source": "Reuters",
                "text": "bad",
                "ticker": "PDD",
                "sentiment_label": "negative",
                "sentiment_score": 0.95,
            }
        ]
    ).write_parquet(sent_path)
    monkeypatch.setattr("src.portfolio.sentiment_sides.PROCESSED_SENTIMENT_PATH", sent_path)

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        """
mode: demo
market:
  tickers: [PDD]
portfolio:
  allow_shorts: false
  sentiment_position_sides: true
""".strip(),
        encoding="utf-8",
    )
    app = load_app_config(run_yaml)
    sides = effective_position_sides(app, ["PDD"])
    assert sides is None or "PDD" not in sides
