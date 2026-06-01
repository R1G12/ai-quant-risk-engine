"""Tests for blended expected returns."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.portfolio.expected_returns import build_expected_returns, sentiment_mean_returns
from src.utils.config import load_app_config


@pytest.fixture
def risk_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "risk_dataset.parquet"
    rows = []
    end = date(2024, 6, 1)
    for i in range(10):
        d = datetime_from_date(end - timedelta(days=i))
        rows.append(
            {
                "timestamp": d,
                "ticker": "AAA",
                "returns": 0.001,
                "bullish_ratio": 0.6,
            }
        )
        rows.append(
            {
                "timestamp": d,
                "ticker": "BBB",
                "returns": 0.002,
                "bullish_ratio": 0.4,
            }
        )
    pl.DataFrame(rows).write_parquet(path)
    monkeypatch.setattr("src.portfolio.expected_returns.RISK_DATASET_PATH", path)
    return path


def datetime_from_date(d: date):
    from datetime import datetime, timezone

    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


@pytest.fixture
def run_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(
        """
mode: demo
market:
  tickers: [AAA, BBB]
portfolio:
  weighting: optimised
  sentiment_mu_blend: 0.5
  sentiment_mu_mode: fixed
  sentiment_mu_scale: 1.0
  sentiment_mu_window_days: 30
  regime_sentiment_mix: { low: 0.5, mid: 1.0, high: 1.0 }
""".strip(),
        encoding="utf-8",
    )
    return p


def test_build_expected_returns_alpha_zero_is_historical(
    risk_dataset: Path, run_yaml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_yaml.write_text(
        run_yaml.read_text(encoding="utf-8").replace("sentiment_mu_blend: 0.5", "sentiment_mu_blend: 0"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.portfolio.sentiment_sides.ticker_sentiment_scores",
        lambda *a, **k: {"AAA": 1.0},
    )
    monkeypatch.setattr("src.portfolio.regime_policy.latest_hmm_regime_label", lambda _a: "high")
    app = load_app_config(run_yaml)
    mu, meta = build_expected_returns(["AAA", "BBB"], app)
    assert meta["mu_mode"] == "historical_only"
    assert mu[0] == pytest.approx(0.001)
    assert mu[1] == pytest.approx(0.002)


def test_build_expected_returns_blend(
    risk_dataset: Path, run_yaml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.portfolio.sentiment_sides.ticker_sentiment_scores",
        lambda *a, **k: {"AAA": 1.0, "BBB": -1.0},
    )
    monkeypatch.setattr("src.portfolio.regime_policy.latest_hmm_regime_label", lambda _a: "mid")
    app = load_app_config(run_yaml)
    mu, meta = build_expected_returns(["AAA", "BBB"], app)
    assert meta["alpha_eff"] == pytest.approx(0.5)
    assert mu[0] != mu[1]


def test_sentiment_mean_returns_vol_scaled() -> None:
    mu = sentiment_mean_returns(
        ["A"],
        load_app_config(),
        window_days=30,
        mode="vol_scaled",
        scale=2.0,
        vol=np.array([0.01]),
    )
    assert mu[0] == 0.0  # no scores mocked


def test_for_min_variance_skips_sentiment(risk_dataset: Path, run_yaml: Path) -> None:
    app = load_app_config(run_yaml)
    mu, meta = build_expected_returns(["AAA", "BBB"], app, for_min_variance=True)
    assert meta["for_min_variance"] is True
    assert meta["mu_mode"] == "historical_only"
