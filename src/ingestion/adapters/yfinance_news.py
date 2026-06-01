"""Yahoo Finance news adapter (per run-profile tickers)."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import polars as pl

from src.ingestion.news_schema import (
    RAW_NEWS_COLUMNS,
    clamp_headline_cap,
    load_ingestion_news_config,
)
from src.market.adapters.yfinance import _make_curl_session
from src.utils.config import load_app_config
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _bootstrap_ssl_certs() -> None:
    """Point curl/requests at certifi bundle when available (helps Windows SSL)."""
    try:
        import certifi

        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
    except ImportError:
        pass


def yfinance_news_strategies() -> list[tuple[str, Any | None]]:
    """Session order tuned for Yahoo **news** (no-verify first, then certifi-backed verify)."""
    _bootstrap_ssl_certs()
    strategies: list[tuple[str, Any | None]] = []

    no_verify = _make_curl_session(verify=False)
    if no_verify is not None:
        strategies.append(("curl_cffi_no_verify", no_verify))

    verify_sess = _make_curl_session(verify=True)
    if verify_sess is not None:
        strategies.append(("curl_cffi", verify_sess))

    if os.getenv("YFINANCE_SSL_VERIFY", "1").lower() not in ("0", "false", "no"):
        strategies.append(("yfinance_default", None))
    elif sys.platform == "win32":
        LOGGER.debug("Skipping yfinance_default on Windows when YFINANCE_SSL_VERIFY=0")
    else:
        strategies.append(("yfinance_default", None))
    return strategies


def _unwrap_news_item(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten yfinance 1.4+ nested ``{'id': ..., 'content': {...}}`` payloads."""
    inner = item.get("content")
    if isinstance(inner, dict):
        return {**inner, **{k: v for k, v in item.items() if k != "content"}}
    return item


def _parse_publish_date(item: dict[str, Any]) -> date | None:
    ts = (
        item.get("providerPublishTime")
        or item.get("pubDate")
        or item.get("published")
        or item.get("displayTime")
    )
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
    if isinstance(ts, datetime):
        return ts.date() if ts.tzinfo is None else ts.astimezone(timezone.utc).date()
    if isinstance(ts, date):
        return ts
    text = str(ts).strip()
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _normalize_item(item: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    item = _unwrap_news_item(item)
    title = item.get("title") or item.get("headline")
    if not title and isinstance(item.get("content"), dict):
        title = item["content"].get("title")
    if not title:
        return None
    title = str(title).strip()
    summary = item.get("summary") or item.get("description") or ""
    if isinstance(item.get("content"), dict):
        summary = summary or item["content"].get("summary") or item["content"].get("description") or ""
    content = str(summary).strip() or title
    pub = _parse_publish_date(item)
    if pub is None:
        return None
    provider = item.get("provider")
    publisher = (
        item.get("publisher")
        or item.get("publisherName")
        or item.get("source")
        or (provider.get("displayName") if isinstance(provider, dict) else None)
        or "yfinance"
    )
    return {
        "date": pub.isoformat(),
        "ticker": ticker.upper(),
        "source": str(publisher),
        "title": title,
        "content": content,
    }


def _raw_news_items(ticker: Any, *, max_headlines: int) -> list[dict[str, Any]]:
    """Call yfinance news API (prefer get_news for explicit count)."""
    raw = ticker.get_news(count=max_headlines, tab="news")
    if not raw:
        return []
    return [item for item in raw if isinstance(item, dict)]


def fetch_ticker_news(
    ticker: str,
    *,
    session: Any = None,
    lookback_days: int = 30,
    max_headlines: int = 40,
) -> list[dict[str, Any]]:
    """Fetch and filter news for one symbol."""
    import yfinance as yf

    t = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
    raw_items = _raw_news_items(t, max_headlines=max_headlines)
    cutoff = date.today() - timedelta(days=lookback_days)
    rows: list[dict[str, Any]] = []
    for item in raw_items:
        row = _normalize_item(item, ticker)
        if row is None:
            continue
        if date.fromisoformat(row["date"]) < cutoff:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:max_headlines]


def fetch_ticker_news_resilient(
    ticker: str,
    *,
    lookback_days: int = 30,
    max_headlines: int = 40,
) -> tuple[list[dict[str, Any]], tuple[str, Any | None] | None]:
    """Fetch news; return rows and (strategy_label, session) when one succeeds."""
    last_error: str | None = None
    for label, session in yfinance_news_strategies():
        LOGGER.info("yfinance news %s: trying strategy=%s", ticker, label)
        try:
            batch = fetch_ticker_news(
                ticker,
                session=session,
                lookback_days=lookback_days,
                max_headlines=max_headlines,
            )
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("yfinance news %s strategy=%s failed: %s", ticker, label, exc)
            continue
        if not batch:
            LOGGER.info("yfinance news %s: strategy=%s returned 0 parseable headlines", ticker, label)
            continue
        LOGGER.info(
            "yfinance news %s: strategy=%s kept %s headlines (max %s)",
            ticker,
            label,
            len(batch),
            max_headlines,
        )
        return batch, (label, session)
    if last_error:
        LOGGER.warning("yfinance news failed for %s (all strategies): %s", ticker, last_error)
    else:
        LOGGER.info("yfinance news %s: kept 0 headlines after all strategies (max %s)", ticker, max_headlines)
    return [], None


def load_yfinance_news() -> pl.DataFrame:
    """Load recent headlines for all tickers in the active run profile."""
    _bootstrap_ssl_certs()
    cfg = load_ingestion_news_config()
    max_h = clamp_headline_cap(cfg["max_headlines_per_ticker"])
    lookback = int(cfg["lookback_days"])
    app = load_app_config()
    tickers = [str(t).strip().upper() for t in app.market.tickers if str(t).strip()]
    if not tickers:
        LOGGER.warning("No market tickers configured for yfinance news ingest")
        return pl.DataFrame(schema={c: pl.Utf8 for c in RAW_NEWS_COLUMNS})

    all_rows: list[dict[str, Any]] = []
    active_strategy: tuple[str, Any | None] | None = None

    for ticker in tickers:
        if active_strategy is not None:
            label, session = active_strategy
            LOGGER.info("yfinance news %s: reusing strategy=%s", ticker, label)
            try:
                batch = fetch_ticker_news(
                    ticker,
                    session=session,
                    lookback_days=lookback,
                    max_headlines=max_h,
                )
            except Exception as exc:
                LOGGER.warning(
                    "yfinance news %s: cached strategy=%s failed (%s); re-probing",
                    ticker,
                    label,
                    exc,
                )
                batch, active_strategy = fetch_ticker_news_resilient(
                    ticker, lookback_days=lookback, max_headlines=max_h
                )
        else:
            batch, active_strategy = fetch_ticker_news_resilient(
                ticker, lookback_days=lookback, max_headlines=max_h
            )
        all_rows.extend(batch)

    if not all_rows:
        LOGGER.warning("yfinance news returned no rows for tickers=%s", tickers)
        return pl.DataFrame(schema={c: pl.Utf8 for c in RAW_NEWS_COLUMNS})

    df = pl.DataFrame(all_rows).select(RAW_NEWS_COLUMNS)
    df = df.with_columns(pl.col("date").str.to_date())
    LOGGER.info("yfinance news ingest total rows=%s tickers=%s", df.height, df["ticker"].n_unique())
    return df
