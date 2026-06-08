"""Tests for 3-day FinBERT panel with forward-fill."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

from src.dashboards.core.signals_loaders import RECENT_SCORE_HISTORY_DAYS, finbert_recent_days_panel


def _daily_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
    )


def _daily_row(ticker: str, d: date, score: float) -> dict[str, object]:
    ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return {
        "ticker": ticker,
        "timestamp": ts,
        "sentiment_score": score,
        "bullish_ratio": 0.6,
        "negative_ratio": 0.2,
        "article_count": 3,
        "avg_confidence": 0.9,
    }


def test_recent_panel_all_fresh_days() -> None:
    end = date(2026, 5, 10)
    days = [end - timedelta(days=i) for i in range(RECENT_SCORE_HISTORY_DAYS - 1, -1, -1)]
    daily = _daily_frame([_daily_row("AAPL", d, 0.5) for d in days])
    panel, stale = finbert_recent_days_panel(daily, ["AAPL"], window_end=end)
    assert panel.height == RECENT_SCORE_HISTORY_DAYS
    assert str(panel.schema["timestamp"]).startswith("Datetime")
    assert panel.to_pandas()["timestamp"].notna().all()
    assert not panel["is_forward_filled"].any()
    assert not stale.filter(pl.col("ticker") == "AAPL")["has_stale"][0]


def test_recent_panel_forward_fills_tail() -> None:
    end = date(2026, 5, 10)
    only = end - timedelta(days=2)
    daily = _daily_frame([_daily_row("INTC", only, -0.2)])
    panel, stale = finbert_recent_days_panel(daily, ["INTC"], window_end=end)
    assert panel.height == RECENT_SCORE_HISTORY_DAYS
    filled = panel.filter(pl.col("is_forward_filled"))
    assert filled.height == 2
    assert filled["score_as_of"].unique().to_list() == [only]
    row = stale.filter(pl.col("ticker") == "INTC").row(0, named=True)
    assert row["has_stale"]
    assert len(row["stale_day_labels"]) == 2


def test_recent_panel_shared_window_across_tickers() -> None:
    end = date(2026, 5, 10)
    daily = _daily_frame(
        [
            _daily_row("AAPL", end, 0.4),
            _daily_row("NVDA", end - timedelta(days=5), 0.1),
        ]
    )
    panel, _ = finbert_recent_days_panel(daily, ["AAPL", "NVDA"], window_end=end)
    aapl_days = panel.filter(pl.col("ticker") == "AAPL")["day"].to_list()
    nvda_days = panel.filter(pl.col("ticker") == "NVDA")["day"].to_list()
    assert aapl_days == nvda_days
