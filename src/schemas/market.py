"""Market data schema constants."""

from __future__ import annotations

MARKET_CLEAN_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

MARKET_FEATURES_COLUMNS: tuple[str, ...] = (
    *MARKET_CLEAN_COLUMNS,
    "returns",
    "log_returns",
)

PARTITION_COLUMNS: tuple[str, ...] = ("year", "month")
