"""Tests for configs/run.yaml loading and merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_app_config, load_run_config, run_profile_env


@pytest.fixture
def run_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(
        """
mode: live
market:
  tickers: [CVX, XLE, QQQ]
portfolio:
  weighting: partial
  allow_shorts: true
  max_gross_per_ticker: 0.5
  anchor_weights:
    CVX: 0.4
  trailing_stops:
    bull_threshold: 0.25
    bear:
      levels: [-0.02, -0.04, -0.06]
      fractions: [0.5, 0.25, 0.25]
research:
  backtest_weight_source: max_sharpe
""".strip(),
        encoding="utf-8",
    )
    return p


def test_load_run_config_live(run_yaml: Path) -> None:
    run = load_run_config(run_yaml)
    assert run is not None
    assert run.mode == "live"
    assert run.market.tickers == ["CVX", "XLE", "QQQ"]
    assert run.portfolio.weighting == "partial"
    assert run.portfolio.anchor_weights["CVX"] == pytest.approx(0.4)
    assert run.portfolio.trailing_stops.bull_threshold == pytest.approx(0.25)
    assert run.portfolio.trailing_stops.bear.levels == (-0.02, -0.04, -0.06)


def test_run_profile_env_live(run_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_SOURCE", raising=False)
    monkeypatch.delenv("NEWS_SOURCE", raising=False)
    run = load_run_config(run_yaml)
    assert run is not None
    env = run_profile_env(run)
    assert env["MARKET_SOURCE"] == "yfinance"
    assert env["NEWS_SOURCE"] == "yfinance"


def test_run_profile_env_force_overrides_shell(run_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_SOURCE", "sample")
    run = load_run_config(run_yaml)
    assert run is not None
    env = run_profile_env(run, force=True)
    assert env["MARKET_SOURCE"] == "yfinance"


def test_load_app_config_merges_run_profile(run_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARKET_SOURCE", raising=False)
    app = load_app_config(run_yaml)
    assert app.run is not None
    assert app.market.tickers == ["CVX", "XLE", "QQQ"]
    assert app.market.source == "yfinance"


def test_default_run_profile_uses_ci_yaml_when_ci_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils.config import _default_run_profile_path

    monkeypatch.delenv("RUN_PROFILE", raising=False)
    monkeypatch.setenv("CI", "true")
    assert _default_run_profile_path().name == "run.ci.yaml"


def test_prepare_uses_run_mode_not_stale_market_source(
    run_yaml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare validates via run.mode even when MARKET_SOURCE=sample is set in shell."""
    monkeypatch.setenv("MARKET_SOURCE", "sample")
    monkeypatch.setattr("src.portfolio.prepare.PROJECT_ROOT", tmp_path)
    calls: list[str] = []

    def _capture(tickers: list[str], source: str):
        calls.append(source)
        from src.market.ticker_validation import TickerFilterResult

        return TickerFilterResult(requested=tickers, valid=list(tickers))

    monkeypatch.setattr("src.portfolio.prepare.validate_market_tickers", _capture)
    from src.utils.config import apply_run_profile_env, load_app_config
    from src.portfolio.prepare import materialize_run

    run = load_run_config(run_yaml)
    assert run is not None
    apply_run_profile_env(run, force=True)
    app = load_app_config(run_yaml)
    app.risk.portfolio.holdings_path = str(tmp_path / "holdings.parquet")
    materialize_run(app)
    assert calls == ["yfinance"]
    assert app.market.source == "yfinance"
    assert app.risk.optimization.weighting == "partial"
    assert app.risk.optimization.allow_shorts is True
    assert app.research.backtest.weight_source == "max_sharpe"
