"""Background yfinance fetch for Streamlit live price toggle."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dataclasses import replace
from datetime import date, timedelta

from src.market.adapters.yfinance import load_yfinance_market
from src.utils.config import AppConfig, MarketConfig, load_app_config
from src.utils.logger import get_logger
from src.utils.paths import MARKET_LIVE_CACHE_DIR, MARKET_LIVE_RAW_PATH, ensure_dir

LOGGER = get_logger(__name__)


@dataclass
class LiveFetchState:
    """Thread-safe live fetch status."""

    ready: bool = False
    error: str | None = None
    path: str = str(MARKET_LIVE_RAW_PATH)


_lock = threading.Lock()
_state = LiveFetchState()
_thread: threading.Thread | None = None


def live_cache_path() -> str:
    return str(MARKET_LIVE_RAW_PATH)


def is_live_ready() -> bool:
    with _lock:
        return _state.ready or MARKET_LIVE_RAW_PATH.is_file()


def live_fetch_error() -> str | None:
    with _lock:
        return _state.error


def market_config_for_live(cfg: MarketConfig) -> MarketConfig:
    """Rolling ~1Y ending today for live dashboard fetch (never future dates)."""
    today = date.today()
    start = today - timedelta(days=cfg.rolling_days)
    return replace(
        cfg,
        start_date=start.isoformat(),
        end_date=today.isoformat(),
    )


def fetch_live_market(cfg: MarketConfig) -> Path:
    """Download OHLCV and write parquet cache."""
    ensure_dir(MARKET_LIVE_CACHE_DIR)
    lf = load_yfinance_market(market_config_for_live(cfg), prefer_period=True)
    df = lf.collect()
    df.write_parquet(MARKET_LIVE_RAW_PATH)
    LOGGER.info("Live market cache written", extra={"rows": df.height, "path": str(MARKET_LIVE_RAW_PATH)})
    return MARKET_LIVE_RAW_PATH


def _run_fetch(app: AppConfig) -> None:
    global _state
    try:
        fetch_live_market(market_config_for_live(app.market))
        with _lock:
            _state = LiveFetchState(ready=True, error=None, path=str(MARKET_LIVE_RAW_PATH))
    except Exception as exc:
        LOGGER.warning("Live market fetch failed: %s", exc)
        with _lock:
            _state = LiveFetchState(ready=False, error=str(exc), path=str(MARKET_LIVE_RAW_PATH))


def start_background_fetch(app: AppConfig | None = None) -> None:
    """Start yfinance download in a daemon thread (no-op if already running/ready)."""
    global _thread, _state
    if MARKET_LIVE_RAW_PATH.is_file():
        with _lock:
            _state = LiveFetchState(ready=True, error=None, path=str(MARKET_LIVE_RAW_PATH))
        return
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
    app = app or load_app_config()
    _thread = threading.Thread(target=_run_fetch, args=(app,), daemon=True, name="live_market_fetch")
    _thread.start()


def load_live_prices(date_range=None) -> pl.DataFrame | None:
    """Load cached live OHLCV; optional DateRange filter applied by caller."""
    if not MARKET_LIVE_RAW_PATH.is_file():
        return None
    return pl.read_parquet(MARKET_LIVE_RAW_PATH)


def fetch_tickers_market(tickers: list[str], cfg: MarketConfig) -> pl.DataFrame:
    """Download OHLCV for arbitrary tickers (Streamlit custom selection)."""
    if not tickers:
        raise ValueError("No tickers provided")
    custom_cfg = replace(cfg, tickers=[t.upper() for t in tickers])
    return load_yfinance_market(market_config_for_live(custom_cfg), prefer_period=True).collect()


def invalidate_live_cache() -> None:
    """Clear live cache and reset fetch state (thread-safe; for Streamlit retry)."""
    global _thread, _state
    with _lock:
        _state = LiveFetchState(ready=False, error=None, path=str(MARKET_LIVE_RAW_PATH))
        _thread = None
        if MARKET_LIVE_RAW_PATH.is_file():
            backup = MARKET_LIVE_RAW_PATH.with_suffix(".parquet.bak")
            if backup.is_file():
                backup.unlink()
            MARKET_LIVE_RAW_PATH.rename(backup)


def reset_live_state_for_tests() -> None:
    """Reset module state (tests only)."""
    global _thread, _state
    with _lock:
        _state = LiveFetchState()
        _thread = None
        if MARKET_LIVE_RAW_PATH.is_file():
            MARKET_LIVE_RAW_PATH.unlink(missing_ok=True)
        backup = MARKET_LIVE_RAW_PATH.with_suffix(".parquet.bak")
        if backup.is_file():
            backup.unlink()
