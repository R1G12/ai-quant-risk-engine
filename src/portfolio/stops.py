"""Sentiment-derived trailing stop configuration (notebook-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Sentiment thresholds (trading_risk_manager_final.ipynb)
SENTIMENT_BULL_THRESHOLD = 0.3
SENTIMENT_BEAR_THRESHOLD = -0.3

BULL_STOP_LEVELS = (-0.08, -0.14, -0.20)
BULL_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)
NEUTRAL_STOP_LEVELS = (-0.05, -0.10, -0.15)
NEUTRAL_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)
BEAR_STOP_LEVELS = (-0.03, -0.06, -0.10)
BEAR_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class TrailingStopSet:
    """Three tranche stops for one ticker."""

    sentiment_score: float
    regime_tag: str
    stops: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]

    @property
    def stop_1(self) -> dict[str, Any]:
        return self.stops[0]

    @property
    def stop_2(self) -> dict[str, Any]:
        return self.stops[1]

    @property
    def stop_3(self) -> dict[str, Any]:
        return self.stops[2]

    @property
    def tranche_1_level(self) -> float:
        """Tightest stop (fires first): highest level, e.g. -5% before -10% / -15%."""
        return float(max(s["level"] for s in self.stops))


def build_stops(
    sentiment_score: float,
    *,
    use_manual: bool = False,
    manual_levels: tuple[float, ...] = (-0.05, -0.10, -0.15),
    manual_fractions: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3),
) -> tuple[list[dict[str, Any]], str]:
    """Return stop dicts and regime tag from a sentiment score (-1 bearish → +1 bullish)."""
    if use_manual:
        levels, fracs, tag = manual_levels, manual_fractions, "Manual"
    elif sentiment_score >= SENTIMENT_BULL_THRESHOLD:
        levels, fracs, tag = BULL_STOP_LEVELS, BULL_STOP_FRACTIONS, "Bullish-derived"
    elif sentiment_score <= SENTIMENT_BEAR_THRESHOLD:
        levels, fracs, tag = BEAR_STOP_LEVELS, BEAR_STOP_FRACTIONS, "Bearish-derived"
    else:
        levels, fracs, tag = NEUTRAL_STOP_LEVELS, NEUTRAL_STOP_FRACTIONS, "Neutral-derived"

    stops = [
        {
            "level": float(level),
            "exit_fraction": float(frac),
            "label": f"Stop {i + 1} ({level:.0%})",
        }
        for i, (level, frac) in enumerate(zip(levels, fracs, strict=True))
    ]
    return stops, tag


def trailing_stop_set(sentiment_score: float, **kwargs: Any) -> TrailingStopSet:
    """Build a TrailingStopSet for one ticker."""
    stops, tag = build_stops(sentiment_score, **kwargs)
    return TrailingStopSet(
        sentiment_score=sentiment_score,
        regime_tag=tag,
        stops=(stops[0], stops[1], stops[2]),
    )
