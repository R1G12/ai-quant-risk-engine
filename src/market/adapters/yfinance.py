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


def yfinance_download_strategies() -> list[tuple[str, object | None]]:
    """Ordered Yahoo download strategies (SSL-friendly order for Windows)."""
    strategies: list[tuple[str, object | None]] = [
        ("yfinance_default", None),
        ("curl_cffi_no_verify", _make_curl_session(verify=False)),
        ("curl_cffi", _yfinance_session()),
    ]
    if os.getenv("YFINANCE_SSL_VERIFY", "1").lower() in ("0", "false", "no"):
        strategies = [("curl_cffi_no_verify", _make_curl_session(verify=False))] + [
            s for s in strategies if s[0] != "curl_cffi_no_verify"
        ]
    return strategies


def download_tickers_history(
    tickers: list[str],
    *,
    session=None,
    period: str = "1y",
    pause_s: float = 0.4,
) -> list[pl.DataFrame]:
    """Download per-ticker history frames (shared by ingest and validation)."""
    return _download_per_ticker_period(tickers, session=session, period=period, pause_s=pause_s)


def tickers_with_yfinance_data(
    tickers: list[str],
    *,
    rolling_days: int = 30,
    pause_s: float = 0.2,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (valid, skipped, reasons) using the same multi-strategy path as ingest."""
    period = period_for_rolling_days(rolling_days)
    valid: list[str] = []
    reasons: dict[str, str] = {}
    remaining = list(tickers)

    for label, session in yfinance_download_strategies():
        if not remaining:
            break
        frames = download_tickers_history(remaining, session=session, period=period, pause_s=pause_s)
        found = {t for f in frames for t in f["ticker"].unique().to_list()}
        if found:
            LOGGER.info(
                "yfinance ticker check strategy=%s validated=%d/%d",
                label,
                len(found),
                len(tickers),
            )
        for t in tickers:
            if t in found and t not in valid:
                valid.append(t)
        remaining = [t for t in remaining if t not in found]

    skipped = [t for t in tickers if t not in valid]
    for t in skipped:
        reasons[t] = "no price history from yfinance (all download strategies failed)"
    return valid, skipped, reasons


def _collect_frames(cfg: MarketConfig, *, prefer_period: bool) -> list[pl.DataFrame]:
    """Try download strategies; merge tickers found across attempts."""
    period = period_for_rolling_days(cfg.rolling_days)
    frames_by_ticker: dict[str, pl.DataFrame] = {}

    for label, session in yfinance_download_strategies():
        remaining = [t for t in cfg.tickers if t not in frames_by_ticker]
        if not remaining:
            break
        attempt = download_tickers_history(remaining, session=session, period=period)
        for f in attempt:
            t = f["ticker"].unique().to_list()[0]
            frames_by_ticker[t] = f
        if not prefer_period and len(frames_by_ticker) < len(cfg.tickers):
            have = set(frames_by_ticker)
            batch_cfg = MarketConfig(
                source=cfg.source,
                tickers=remaining,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                rolling_days=cfg.rolling_days,
            )
            for bf in _download_batch(batch_cfg, session=session):
                t = bf["ticker"][0]
                if t not in have:
                    frames_by_ticker[t] = bf
                    have.add(t)
        if attempt:
            LOGGER.info(
                "yfinance strategy=%s tickers=%d/%d",
                label,
                len(frames_by_ticker),
                len(cfg.tickers),
            )

    return list(frames_by_ticker.values())


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

    ingested = {t for f in frames for t in f["ticker"].unique().to_list()}
    missing = [t for t in cfg.tickers if t not in ingested]
    if missing:
        LOGGER.warning(
            "yfinance returned no rows for %d ticker(s): %s",
            len(missing),
            missing,
        )
    if len(ingested) < 2:
        raise ValueError(
            f"yfinance ingest: need at least 2 tickers with data, got {len(ingested)} "
            f"({sorted(ingested)}). Missing: {missing}"
        )

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


def _pandas_ohlcv_to_polars(part, ticker: str) -> pl.DataFrame:
    part = part.reset_index()
    part.columns = [str(c).lower() for c in part.columns]
    part = part.rename(columns={"date": "timestamp", "adj close": "adj_close"})
    return pl.from_pandas(part).with_columns(pl.lit(ticker).alias("ticker"))


def _parse_yfinance_frames(pdf, tickers: list[str]) -> list[pl.DataFrame]:
    if pdf is None or pdf.empty:
        return []
    frames: list[pl.DataFrame] = []
    if len(tickers) == 1:
        frames.append(_pandas_ohlcv_to_polars(pdf, tickers[0]))
        return frames
    for ticker in tickers:
        if not hasattr(pdf.columns, "get_level_values") or ticker not in pdf.columns.get_level_values(0):
            continue
        frames.append(_pandas_ohlcv_to_polars(pdf[ticker], ticker))
    return frames
