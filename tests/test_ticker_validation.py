"""Ticker validation and sample no-synthetic policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.market.adapters.sample import filter_sample_tickers, load_sample_market
from src.market.ticker_validation import TickerFilterResult, validate_market_tickers
from src.utils.config import MarketConfig


def test_validate_sample_skips_unknown_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.market.ticker_validation._tickers_in_sample_bundle",
        lambda: {"AAPL", "MSFT", "GS"},
    )
    result = validate_market_tickers(["AAPL", "XLU2", "MSFT"], "sample")
    assert result.valid == ["AAPL", "MSFT"]
    assert result.skipped == ["XLU2"]
    assert "bundled sample" in result.reasons["XLU2"]


def test_require_min_tickers_raises() -> None:
    from src.market.ticker_validation import require_min_tickers

    result = TickerFilterResult(requested=["XLU2"], valid=[], skipped=["XLU2"])
    with pytest.raises(ValueError, match="at least 2"):
        require_min_tickers(result, context="test")


def test_validate_yfinance_uses_fallback_strategies(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_strategies():
        return [("mock_a", None), ("mock_b", "session_ok")]

    def fake_download(tickers, *, session=None, period="1y", pause_s=0.4):
        calls.append(str(session))
        if session is None:
            return []
        import polars as pl

        return [
            pl.DataFrame(
                {
                    "timestamp": ["2024-01-02"],
                    "ticker": [t],
                    "close": [100.0],
                }
            )
            for t in tickers
        ]

    monkeypatch.setattr(
        "src.market.adapters.yfinance.yfinance_download_strategies",
        fake_strategies,
    )
    monkeypatch.setattr(
        "src.market.adapters.yfinance.download_tickers_history",
        fake_download,
    )
    result = validate_market_tickers(["CVX", "QQQ"], "yfinance")
    assert result.valid == ["CVX", "QQQ"]
    assert calls == ["None", "session_ok"]


def test_load_sample_market_no_synthetic_for_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    from src.market import ticker_validation

    bundle = tmp_path / "sample"
    bundle.mkdir()
    pl.DataFrame(
        {
            "timestamp": ["2024-01-02"],
            "ticker": ["AAPL"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
            "source": ["sample"],
        }
    ).write_parquet(bundle / "a.parquet")

    monkeypatch.setattr(ticker_validation, "EXTERNAL_SAMPLE_MARKET_DIR", bundle)
    monkeypatch.setattr(
        "src.market.adapters.sample.EXTERNAL_SAMPLE_MARKET_DIR",
        bundle,
    )

    cfg = MarketConfig(
        tickers=["AAPL", "FAKE"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    with pytest.raises(ValueError, match="at least 2"):
        load_sample_market(cfg)
