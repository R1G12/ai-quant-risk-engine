"""Validate market tickers before prepare/ingest (skip invalid, never synthesize them)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.utils.paths import EXTERNAL_SAMPLE_MARKET_DIR

MIN_TICKERS = 2


@dataclass
class TickerFilterResult:
    """Tickers that passed validation vs skipped with reasons."""

    requested: list[str]
    valid: list[str]
    skipped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.valid) >= MIN_TICKERS


def format_skip_messages(result: TickerFilterResult) -> list[str]:
    """Human-readable lines for CLI / logs."""
    if not result.skipped:
        return []
    lines = [
        f"Skipped {len(result.skipped)} ticker(s) — using {len(result.valid)} of "
        f"{len(result.requested)} requested:"
    ]
    for t in result.skipped:
        lines.append(f"  - {t}: {result.reasons.get(t, 'unavailable')}")
    return lines


def _validate_yfinance(tickers: list[str]) -> TickerFilterResult:
    """Require recent Yahoo history using the same multi-strategy path as market ingest."""
    from src.market.adapters.yfinance import tickers_with_yfinance_data

    valid, skipped, reasons = tickers_with_yfinance_data(tickers, rolling_days=30)
    return TickerFilterResult(requested=list(tickers), valid=valid, skipped=skipped, reasons=reasons)


def _tickers_in_sample_bundle() -> set[str]:
    if not EXTERNAL_SAMPLE_MARKET_DIR.is_dir():
        return set()
    files = list(EXTERNAL_SAMPLE_MARKET_DIR.glob("**/*.parquet"))
    if not files:
        return set()
    import polars as pl

    return set(
        pl.scan_parquet(str(EXTERNAL_SAMPLE_MARKET_DIR / "**/*.parquet"))
        .select("ticker")
        .unique()
        .collect()
        .get_column("ticker")
        .to_list()
    )


def _validate_sample_bundle(tickers: list[str]) -> TickerFilterResult:
    """Sample mode: only tickers present in bundled parquet (no synthetic fill-in)."""
    available = _tickers_in_sample_bundle()
    valid: list[str] = []
    skipped: list[str] = []
    reasons: dict[str, str] = {}

    for ticker in tickers:
        if ticker in available:
            valid.append(ticker)
        else:
            skipped.append(ticker)
            if not available:
                reasons[ticker] = "no bundled sample data on disk"
            else:
                reasons[ticker] = "not in bundled sample dataset (synthetic fill-in disabled)"

    return TickerFilterResult(requested=list(tickers), valid=valid, skipped=skipped, reasons=reasons)


def validate_market_tickers(tickers: list[str], source: str) -> TickerFilterResult:
    """Filter tickers by data source. Set AQRE_SKIP_TICKER_VALIDATION=1 to bypass (tests)."""
    if os.getenv("AQRE_SKIP_TICKER_VALIDATION", "").lower() in ("1", "true", "yes"):
        return TickerFilterResult(requested=list(tickers), valid=list(tickers))

    normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if source == "yfinance":
        return _validate_yfinance(normalized)
    return _validate_sample_bundle(normalized)


def require_min_tickers(result: TickerFilterResult, *, context: str = "pipeline") -> None:
    """Raise if too few tickers remain after filtering."""
    if result.ok:
        return
    detail = "; ".join(f"{t} ({result.reasons.get(t, '?')})" for t in result.skipped[:8])
    extra = f" …and {len(result.skipped) - 8} more" if len(result.skipped) > 8 else ""
    raise ValueError(
        f"{context}: need at least {MIN_TICKERS} valid tickers after validation, "
        f"got {len(result.valid)}. Skipped: {detail}{extra}"
    )
