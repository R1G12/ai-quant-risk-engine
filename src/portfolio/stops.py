"""Sentiment-derived trailing stop configuration (notebook-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.config import AppConfig

# Defaults aligned with trading_risk_manager_final.ipynb
SENTIMENT_BULL_THRESHOLD = 0.3
SENTIMENT_BEAR_THRESHOLD = -0.3

BULL_STOP_LEVELS = (-0.08, -0.14, -0.20)
BULL_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)
NEUTRAL_STOP_LEVELS = (-0.05, -0.10, -0.15)
NEUTRAL_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)
BEAR_STOP_LEVELS = (-0.03, -0.06, -0.10)
BEAR_STOP_FRACTIONS = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class TrailingStopTranchePolicy:
    """Three drawdown levels and exit fractions per tranche."""

    levels: tuple[float, float, float]
    fractions: tuple[float, float, float]


@dataclass(frozen=True)
class TrailingStopPolicy:
    """Full trailing-stop policy (from configs/run.yaml or defaults)."""

    bull_threshold: float = SENTIMENT_BULL_THRESHOLD
    bear_threshold: float = SENTIMENT_BEAR_THRESHOLD
    bull: TrailingStopTranchePolicy = TrailingStopTranchePolicy(BULL_STOP_LEVELS, BULL_STOP_FRACTIONS)
    neutral: TrailingStopTranchePolicy = TrailingStopTranchePolicy(
        NEUTRAL_STOP_LEVELS, NEUTRAL_STOP_FRACTIONS
    )
    bear: TrailingStopTranchePolicy = TrailingStopTranchePolicy(BEAR_STOP_LEVELS, BEAR_STOP_FRACTIONS)
    use_manual: bool = False
    manual: TrailingStopTranchePolicy = TrailingStopTranchePolicy(
        NEUTRAL_STOP_LEVELS, NEUTRAL_STOP_FRACTIONS
    )


DEFAULT_TRAILING_STOP_POLICY = TrailingStopPolicy()


def _parse_tranche(raw: dict[str, Any] | None, default: TrailingStopTranchePolicy) -> TrailingStopTranchePolicy:
    if not raw:
        return default
    levels = raw.get("levels")
    fractions = raw.get("fractions")
    if levels is None and fractions is None:
        return default
    lv = tuple(float(x) for x in (levels if levels is not None else default.levels))
    fr = tuple(float(x) for x in (fractions if fractions is not None else default.fractions))
    if len(lv) != 3 or len(fr) != 3:
        raise ValueError("trailing_stops tranche requires exactly 3 levels and 3 fractions")
    return TrailingStopTranchePolicy((lv[0], lv[1], lv[2]), (fr[0], fr[1], fr[2]))


def trailing_stop_policy_from_dict(raw: dict[str, Any] | None) -> TrailingStopPolicy:
    """Parse portfolio.trailing_stops from configs/run.yaml."""
    if not raw:
        return DEFAULT_TRAILING_STOP_POLICY
    base = DEFAULT_TRAILING_STOP_POLICY
    manual_block = raw.get("manual") if isinstance(raw.get("manual"), dict) else None
    return TrailingStopPolicy(
        bull_threshold=float(raw.get("bull_threshold", base.bull_threshold)),
        bear_threshold=float(raw.get("bear_threshold", base.bear_threshold)),
        bull=_parse_tranche(raw.get("bull"), base.bull),
        neutral=_parse_tranche(raw.get("neutral"), base.neutral),
        bear=_parse_tranche(raw.get("bear"), base.bear),
        use_manual=bool(raw.get("use_manual", False)),
        manual=_parse_tranche(manual_block, base.manual),
    )


def resolve_trailing_stop_policy(app: AppConfig | None) -> TrailingStopPolicy:
    """Active policy from run profile, else module defaults."""
    if app is not None and app.run is not None:
        return app.run.portfolio.trailing_stops
    return DEFAULT_TRAILING_STOP_POLICY


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


def _stops_from_tranche(tranche: TrailingStopTranchePolicy) -> list[dict[str, Any]]:
    return [
        {
            "level": float(level),
            "exit_fraction": float(frac),
            "label": f"Stop {i + 1} ({level:.0%})",
        }
        for i, (level, frac) in enumerate(zip(tranche.levels, tranche.fractions, strict=True))
    ]


def build_stops(
    sentiment_score: float,
    policy: TrailingStopPolicy | None = None,
    *,
    use_manual: bool | None = None,
    manual_levels: tuple[float, ...] | None = None,
    manual_fractions: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return stop dicts and regime tag from a sentiment score (-1 bearish → +1 bullish)."""
    policy = policy or DEFAULT_TRAILING_STOP_POLICY
    manual = use_manual if use_manual is not None else policy.use_manual

    if manual:
        tranche = policy.manual
        if manual_levels is not None or manual_fractions is not None:
            lv = manual_levels or tranche.levels
            fr = manual_fractions or tranche.fractions
            tranche = TrailingStopTranchePolicy((lv[0], lv[1], lv[2]), (fr[0], fr[1], fr[2]))
        stops, tag = _stops_from_tranche(tranche), "Manual"
    elif sentiment_score >= policy.bull_threshold:
        stops, tag = _stops_from_tranche(policy.bull), "Bullish-derived"
    elif sentiment_score <= policy.bear_threshold:
        stops, tag = _stops_from_tranche(policy.bear), "Bearish-derived"
    else:
        stops, tag = _stops_from_tranche(policy.neutral), "Neutral-derived"

    return stops, tag


def trailing_stop_set(
    sentiment_score: float,
    policy: TrailingStopPolicy | None = None,
    **kwargs: Any,
) -> TrailingStopSet:
    """Build a TrailingStopSet for one ticker."""
    stops, tag = build_stops(sentiment_score, policy, **kwargs)
    return TrailingStopSet(
        sentiment_score=sentiment_score,
        regime_tag=tag,
        stops=(stops[0], stops[1], stops[2]),
    )
