"""yfinance market data adapter (pandas boundary – convert immediately to Polars)."""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import polars as pl

from src.utils.config import MarketConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_YFINANCE_ERROR_HINT = (
    "yfinance returned no data. Try: pip install -U 'yfinance>=1.3.0' 'curl_cffi>=0.15', "
    "check network/DNS (fc.yahoo.com), SSL certs, or retry later (Yahoo rate limits). "
    "If you see SSL certificate errors on Windows, run: pip install certifi"
)


def _make_curl_session(*, verify: bool = True):
    """Create curl_cffi session with browser impersonation."""
    Session = None
    try:
        from curl_cffi.requests import Session
    except ImportError:
        try:
            from curl_cffi import requests as curl_requests

            Session = curl_requests.Session
        except ImportError:
            return None

    if Session is None:
        return None

    for impersonate in ("chrome", "chrome124", "chrome110", "edge99", "safari15_5"):
        try:
            return Session(impersonate=impersonate, timeout=30, verify=verify)
        except Exception:
            continue
    try:
        return Session(timeout=30, verify=verify)
    except Exception:
        return None


def _yfinance_session():
    """Browser-like session for Yahoo bot protection (TLS fingerprinting)."""
    return _make_curl_session(verify=True)


def period_for_rolling_days(rolling_days: int) -> str:
    """Map rolling window to yfinance period token."""
    if rolling_days <= 5:
        return "5d"
    if rolling_days <= 30:
        return "1mo"
    if rolling_days <= 90:
        return "3mo"
    if rolling_days <= 180:
        return "6mo"
    if rolling_days <= 365:
        return "1y"
    if rolling_days <= 730:
        return "2y"
    return "5y"


def clamp_download_dates(start_date: str, end_date: str, *, today: date | None = None) -> tuple[str, str]:
    """Clamp to real calendar dates; yfinance ``end`` is exclusive."""
    today = today or date.today()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    end = min(end, today)
    if start > end:
        start = end - timedelta(days=365)
    end_exclusive = (end + timedelta(days=1)).isoformat()
    return start.isoformat(), end_exclusive


def _pandas_history_to_frame(hist, ticker: str) -> pl.DataFrame | None:
    if hist is None or hist.empty:
        return None
    part = hist.reset_index()
    part.columns = [str(c).lower() for c in part.columns]
    rename = {"date": "timestamp", "adj close": "adj_close"}
    part = part.rename(columns={k: v for k, v in rename.items() if k in part.columns})
    df = pl.from_pandas(part).with_columns(pl.lit(ticker).alias("ticker"))
    # yfinance extras (dividends, splits, capital gains) differ by ticker — drop for concat
    keep = ["timestamp", "open", "high", "low", "close", "volume", "adj_close", "ticker"]
    return df.select([c for c in keep if c in df.columns])


def _download_per_ticker_period(
    tickers: list[str],
    *,
    session=None,
    period: str = "1y",
    pause_s: float = 0.4,
) -> list[pl.DataFrame]:
    """Per-ticker history — most reliable under Yahoo bot protection."""
    import yfinance as yf

    frames: list[pl.DataFrame] = []
    errors: list[str] = []
    for i, ticker in enumerate(tickers):
        if i > 0 and pause_s > 0:
            time.sleep(pause_s)
        t = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)
        try:
            hist = t.history(period=period, auto_adjust=True, raise_errors=True)
        except TypeError:
            hist = t.history(period=period, auto_adjust=True)
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")
            LOGGER.warning("yfinance history failed for %s: %s", ticker, exc)
            continue
        frame = _pandas_history_to_frame(hist, ticker)
        if frame is not None:
            frames.append(frame)
    if not frames and errors:
        LOGGER.warning("All per-ticker downloads failed: %s", "; ".join(errors[:3]))
    return frames


def _download_batch(
    cfg: MarketConfig,
    *,
    session=None,
) -> list[pl.DataFrame]:
    import yfinance as yf

    start_dl, end_dl = clamp_download_dates(cfg.start_date, cfg.end_date)
    kwargs = dict(
        tickers=cfg.tickers,
        start=start_dl,
        end=end_dl,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if session is not None:
        kwargs["session"] = session
    try:
        pdf = yf.download(**kwargs)
    except Exception as exc:
        LOGGER.warning("yfinance batch download failed: %s", exc)
        return []
    return _parse_yfinance_frames(pdf, cfg.tickers)


def _collect_frames(cfg: MarketConfig, *, prefer_period: bool) -> list[pl.DataFrame]:
    """Try several download strategies until one returns data."""
    period = period_for_rolling_days(cfg.rolling_days)
    strategies: list[tuple[str, object | None]] = [
        ("curl_cffi", _yfinance_session()),
        ("yfinance_default", None),
        # Last resort for Windows SSL store issues (common with curl_cffi)
        ("curl_cffi_no_verify", _make_curl_session(verify=False)),
    ]
    if os.getenv("YFINANCE_SSL_VERIFY", "1").lower() in ("0", "false", "no"):
        # Prefer insecure path first when explicitly requested (dev only)
        strategies = [("curl_cffi_no_verify", _make_curl_session(verify=False))] + strategies[:-1]

    for label, session in strategies:
        attempt = _download_per_ticker_period(cfg.tickers, session=session, period=period)
        if not prefer_period and len(attempt) < len(cfg.tickers):
            have = {t for f in attempt for t in f["ticker"].unique().to_list()}
            for bf in _download_batch(cfg, session=session):
                t = bf["ticker"][0]
                if t not in have:
                    attempt.append(bf)
                    have.add(t)
        if attempt:
            LOGGER.info("yfinance succeeded via strategy=%s rows=%s", label, sum(f.height for f in attempt))
            return attempt
        LOGGER.warning("yfinance strategy=%s returned no rows", label)
    return []


def load_yfinance_market(cfg: MarketConfig, *, prefer_period: bool = False) -> pl.LazyFrame:
    """Download OHLCV via yfinance and return a Polars LazyFrame.

    ``prefer_period=True`` uses per-ticker ``history(period=...)`` only (best for live dashboard).
    """
    LOGGER.info(
        "Downloading market data via yfinance",
        extra={
            "tickers": cfg.tickers,
            "prefer_period": prefer_period,
            "period": period_for_rolling_days(cfg.rolling_days),
        },
    )
    frames = _collect_frames(cfg, prefer_period=prefer_period)

    if not frames:
        raise ValueError(_YFINANCE_ERROR_HINT)

    df = pl.concat(frames, how="vertical_relaxed")
    for src, dst in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume")]:
        if src in df.columns and src != dst:
            df = df.rename({src: dst})

    df = df.with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
        pl.lit("yfinance").alias("source"),
    )
    select_cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "source"]
    if "adj_close" in df.columns:
        select_cols.insert(6, "adj_close")
    return df.select([c for c in select_cols if c in df.columns]).lazy()


def _parse_yfinance_frames(pdf, tickers: list[str]) -> list[pl.DataFrame]:
    if pdf is None or pdf.empty:
        return []
    frames: list[pl.DataFrame] = []
    if len(tickers) == 1:
        ticker = tickers[0]
        part = pdf.reset_index()
        part.columns = [str(c).lower() for c in part.columns]
        part = part.rename(columns={"date": "timestamp", "adj close": "adj_close"})
        part = part.with_columns(pl.lit(ticker).alias("ticker"))
        frames.append(pl.from_pandas(part))
        return frames
    for ticker in tickers:
        if not hasattr(pdf.columns, "get_level_values") or ticker not in pdf.columns.get_level_values(0):
            continue
        part = pdf[ticker].reset_index()
        part.columns = [str(c).lower() for c in part.columns]
        part = part.rename(columns={"date": "timestamp", "adj close": "adj_close"})
        part = part.with_columns(pl.lit(ticker).alias("ticker"))
        frames.append(pl.from_pandas(part))
    return frames
