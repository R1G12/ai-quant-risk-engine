"""Mark prices for tracker positions (processed parquet first, safe yfinance fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import polars as pl

from src.market.adapters.yfinance import (
    download_symbol_history,
    yfinance_close_series,
    yfinance_scalar_float,
)
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.paths import market_processed_glob

LOGGER = get_logger(__name__)

MarkSource = Literal["processed", "yfinance", "missing"]


@dataclass(frozen=True)
class MarkPriceInfo:
    """Latest mark for a ticker with provenance."""

    price: float | None
    as_of_date: date | None
    source: MarkSource


def _processed_market_available() -> bool:
    pattern = market_processed_glob()
    root = Path(pattern.split("*")[0]).parent.parent
    return root.is_dir()


_EMPTY_PANEL = pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8, "close": pl.Float64})


def _close_panel_processed(
    tickers: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    """Daily closes from processed market parquet only."""
    if not tickers or not _processed_market_available():
        return _EMPTY_PANEL.clone()

    try:
        raw = pl.scan_parquet(market_processed_glob()).collect()
        if raw.is_empty() or "timestamp" not in raw.columns:
            return _EMPTY_PANEL.clone()
        df = (
            raw.filter(pl.col("ticker").is_in(tickers))
            .filter(pl.col("timestamp").dt.date() >= start)
            .filter(pl.col("timestamp").dt.date() <= end)
            .sort("timestamp")
            .select(
                pl.col("timestamp").dt.date().alias("date"),
                "ticker",
                pl.col("close").cast(pl.Float64),
            )
        )
    except Exception as exc:
        LOGGER.warning("close_panel processed load failed: %s", exc)
        return _EMPTY_PANEL.clone()

    if df.is_empty():
        return df
    return df.unique(subset=["date", "ticker"], keep="last")


def _yfinance_ticker_closes(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Daily closes for one ticker via yfinance (Tracker fallback)."""
    if start > end:
        return _EMPTY_PANEL.clone()
    hist = download_symbol_history(ticker, start, end)
    series = yfinance_close_series(hist)
    if series is None or series.empty:
        return _EMPTY_PANEL.clone()
    dates = [d.date() if hasattr(d, "date") else d for d in series.index]
    closes = [yfinance_scalar_float(v) for v in series.tolist()]
    return pl.DataFrame({"date": dates, "ticker": [ticker] * len(dates), "close": closes})


def close_panel(
    tickers: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    """Daily closes: processed parquet first, yfinance fallback per missing ticker."""
    if not tickers:
        return _EMPTY_PANEL.clone()

    processed = _close_panel_processed(tickers, start, end)
    parts: list[pl.DataFrame] = []

    for ticker in tickers:
        proc_sub = processed.filter(pl.col("ticker") == ticker) if processed.height else _EMPTY_PANEL.clone()
        if proc_sub.is_empty():
            yf = _yfinance_ticker_closes(ticker, start, end)
            if yf.height:
                parts.append(yf)
            continue

        parts.append(proc_sub)
        proc_max = proc_sub["date"].max()
        if proc_max is not None and proc_max < end:
            gap_start = proc_max + timedelta(days=1)
            yf = _yfinance_ticker_closes(ticker, gap_start, end)
            if yf.height:
                parts.append(yf)

    if not parts:
        return _EMPTY_PANEL.clone()

    return (
        pl.concat(parts)
        .unique(subset=["date", "ticker"], keep="first")
        .sort("date", "ticker")
    )


def _yfinance_last_close(ticker: str, *, lookback_days: int = 7) -> tuple[float | None, date | None]:
    """Last valid close in recent window; end=yesterday to avoid empty today bar."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    hist = download_symbol_history(ticker, start, end)
    series = yfinance_close_series(hist)
    if series is None or series.empty:
        return None, None

    last_idx = series.index[-1]
    as_of = last_idx.date() if hasattr(last_idx, "date") else last_idx
    return yfinance_scalar_float(series.iloc[-1]), as_of


def _latest_from_panel(panel: pl.DataFrame, ticker: str) -> tuple[float | None, date | None]:
    sub = panel.filter(pl.col("ticker") == ticker).sort("date")
    if sub.is_empty():
        return None, None
    row = sub.row(-1, named=True)
    return float(row["close"]), row["date"]


def latest_mark_prices_with_info(
    tickers: list[str],
    app: AppConfig,
    *,
    lookback_days: int = 7,
) -> dict[str, MarkPriceInfo]:
    """Resolve latest close per ticker; processed first, then yfinance lookback."""
    if not tickers:
        return {}

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=max(lookback_days, 30))
    panel = close_panel(tickers, start, end)

    out: dict[str, MarkPriceInfo] = {}
    for ticker in tickers:
        price, as_of = _latest_from_panel(panel, ticker)
        if price is not None and as_of is not None:
            out[ticker] = MarkPriceInfo(price=price, as_of_date=as_of, source="processed")
            continue

        price, as_of = _yfinance_last_close(ticker, lookback_days=lookback_days)
        if price is not None and as_of is not None:
            out[ticker] = MarkPriceInfo(price=price, as_of_date=as_of, source="yfinance")
        else:
            out[ticker] = MarkPriceInfo(price=None, as_of_date=None, source="missing")

    return out


def latest_mark_prices(tickers: list[str], app: AppConfig) -> dict[str, float]:
    """Backward-compatible dict of ticker -> price (skips missing)."""
    info = latest_mark_prices_with_info(tickers, app)
    return {t: m.price for t, m in info.items() if m.price is not None}


def price_history(
    ticker: str,
    app: AppConfig,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Daily close history for charting (processed first)."""
    end = end or (date.today() - timedelta(days=1))
    start = start or (end - timedelta(days=365))

    panel = close_panel([ticker], start, end)
    sub = panel.filter(pl.col("ticker") == ticker).select("date", "close").sort("date")
    if not sub.is_empty():
        return sub

    hist = download_symbol_history(ticker, start, end)
    series = yfinance_close_series(hist)
    if series is None or series.empty:
        price, as_of = _yfinance_last_close(ticker, lookback_days=(end - start).days + 7)
        if price is not None and as_of is not None:
            return pl.DataFrame({"date": [as_of], "close": [price]})
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})

    dates = [d.date() if hasattr(d, "date") else d for d in series.index]
    closes = [yfinance_scalar_float(v) for v in series.tolist()]
    return pl.DataFrame({"date": dates, "close": closes})
