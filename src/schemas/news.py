"""News and sentiment schema constants."""

from __future__ import annotations

SENTIMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "sentiment",
    "confidence",
)
