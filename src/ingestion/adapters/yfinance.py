"""Yahoo Finance news adapter (live mode)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import polars as pl

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

RAW_COLUMNS = ("date", "source", "title", "content", "ticker")


def _nested_content(item: Mapping[str, Any]) -> Mapping[str, Any]:
    content = item.get("content")
    return content if isinstance(content, dict) else {}


def _extract_title(item: Mapping[str, Any]) -> str:
    nested = _nested_content(item)
    title = nested.get("title") or item.get("title") or ""
    return str(title).strip()


def _extract_body(item: Mapping[str, Any], title: str) -> str:
    nested = _nested_content(item)
    for key in ("summary", "description", "body"):
        val = nested.get(key) or item.get(key)
        if val:
            return str(val).strip()
    return title


def _extract_source(item: Mapping[str, Any]) -> str:
    nested = _nested_content(item)
    for key in ("publisher", "provider", "source"):
        val = nested.get(key) or item.get(key)
        if val:
            return str(val).strip()
    return "yfinance"


def _extract_date(item: Mapping[str, Any]) -> str:
    nested = _nested_content(item)
    ts = (
        item.get("providerPublishTime")
        or nested.get("pubDate")
        or nested.get("displayTime")
        or item.get("pubDate")
    )
    if ts is None:
        return datetime.now(timezone.utc).date().isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
    text = str(ts).strip()
    if text.isdigit():
        return datetime.fromtimestamp(float(text), tz=timezone.utc).date().isoformat()
    return text[:10] if len(text) >= 10 else text


def _normalize_item(item: Mapping[str, Any], ticker: str) -> dict[str, str] | None:
    title = _extract_title(item)
    if not title:
        return None
    content = _extract_body(item, title)
    return {
        "date": _extract_date(item),
        "source": _extract_source(item),
        "title": title,
        "content": content or title,
        "ticker": ticker.upper(),
    }


def _fetch_ticker_news(ticker: str, max_per: int) -> list[dict[str, str]]:
    import yfinance as yf

    raw = yf.Ticker(ticker).news or []
    rows: list[dict[str, str]] = []
    for item in raw[:max_per]:
        if not isinstance(item, dict):
            continue
        row = _normalize_item(item, ticker)
        if row:
            rows.append(row)
    return rows


def load_yfinance_news(
    tickers: list[str],
    *,
    max_headlines_per_ticker: int = 10,
) -> pl.DataFrame:
    """Fetch recent headlines per ticker via yfinance."""
    if not tickers:
        return pl.DataFrame(schema={col: pl.Utf8 for col in RAW_COLUMNS})

    all_rows: list[dict[str, str]] = []
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if not symbol:
            continue
        try:
            batch = _fetch_ticker_news(symbol, max_headlines_per_ticker)
            all_rows.extend(batch)
            LOGGER.info(
                "Fetched yfinance news",
                extra={"ticker": symbol, "rows": len(batch)},
            )
        except Exception as exc:
            LOGGER.warning(
                "yfinance news fetch failed for %s: %s",
                symbol,
                exc,
            )

    if not all_rows:
        return pl.DataFrame(schema={col: pl.Utf8 for col in RAW_COLUMNS})

    df = pl.DataFrame(all_rows)
    return df.unique(subset=["ticker", "title", "date"], keep="first")
