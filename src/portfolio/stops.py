"""Trailing stop policy: static % levels or vol-scaled distances with sentiment multipliers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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

DEFAULT_TRANCHE_SIGMAS = (1.5, 2.5, 3.5)
DEFAULT_HORIZON_DAYS = 7
DEFAULT_SENTIMENT_VOL_MULT = {"bull": 1.25, "neutral": 1.0, "bear": 0.75}
DEFAULT_MIN_LEVEL = -0.30
DEFAULT_MAX_LEVEL = -0.008
DEFAULT_FALLBACK_DAILY_VOL = 0.02

TrailingStopMode = Literal["static", "vol_scaled"]


@dataclass(frozen=True)
class TrailingStopTranchePolicy:
    """Three drawdown levels and exit fractions per tranche."""

    levels: tuple[float, float, float]
    fractions: tuple[float, float, float]


@dataclass(frozen=True)
class TrailingStopPolicy:
    """Full trailing-stop policy (from configs/run.yaml or defaults)."""

    mode: TrailingStopMode = "static"
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
    tranche_sigmas: tuple[float, float, float] = DEFAULT_TRANCHE_SIGMAS
    horizon_days: int = DEFAULT_HORIZON_DAYS
    sentiment_vol_mult: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SENTIMENT_VOL_MULT)
    )
    min_level: float = DEFAULT_MIN_LEVEL
    max_level: float = DEFAULT_MAX_LEVEL
    fallback_daily_vol: float = DEFAULT_FALLBACK_DAILY_VOL


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


def _parse_sigmas(raw: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if raw is None:
        return default
    sigmas = tuple(float(x) for x in raw)
    if len(sigmas) != 3:
        raise ValueError("trailing_stops tranche_sigmas requires exactly 3 values")
    return (sigmas[0], sigmas[1], sigmas[2])


def _parse_sentiment_vol_mult(raw: Any, default: dict[str, float]) -> dict[str, float]:
    if not raw or not isinstance(raw, dict):
        return dict(default)
    out = dict(default)
    for key in ("bull", "neutral", "bear"):
        if key in raw:
            out[key] = float(raw[key])
    return out


def trailing_stop_policy_from_dict(raw: dict[str, Any] | None) -> TrailingStopPolicy:
    """Parse portfolio.trailing_stops from configs/run.yaml."""
    if not raw:
        return DEFAULT_TRAILING_STOP_POLICY
    base = DEFAULT_TRAILING_STOP_POLICY
    manual_block = raw.get("manual") if isinstance(raw.get("manual"), dict) else None
    mode = str(raw.get("mode", base.mode)).lower()
    if mode not in ("static", "vol_scaled"):
        raise ValueError(f"trailing_stops mode must be 'static' or 'vol_scaled', got {mode!r}")
    return TrailingStopPolicy(
        mode=mode,  # type: ignore[arg-type]
        bull_threshold=float(raw.get("bull_threshold", base.bull_threshold)),
        bear_threshold=float(raw.get("bear_threshold", base.bear_threshold)),
        bull=_parse_tranche(raw.get("bull"), base.bull),
        neutral=_parse_tranche(raw.get("neutral"), base.neutral),
        bear=_parse_tranche(raw.get("bear"), base.bear),
        use_manual=bool(raw.get("use_manual", False)),
        manual=_parse_tranche(manual_block, base.manual),
        tranche_sigmas=_parse_sigmas(raw.get("tranche_sigmas"), base.tranche_sigmas),
        horizon_days=int(raw.get("horizon_days", base.horizon_days)),
        sentiment_vol_mult=_parse_sentiment_vol_mult(raw.get("sentiment_vol_mult"), base.sentiment_vol_mult),
        min_level=float(raw.get("min_level", base.min_level)),
        max_level=float(raw.get("max_level", base.max_level)),
        fallback_daily_vol=float(raw.get("fallback_daily_vol", base.fallback_daily_vol)),
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
    daily_vol: float | None = None
    vol_scale: float | None = None

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


def _stops_from_levels_and_fractions(
    levels: tuple[float, float, float],
    fractions: tuple[float, float, float],
) -> list[dict[str, Any]]:
    return [
        {
            "level": float(level),
            "exit_fraction": float(frac),
            "label": f"Stop {i + 1} ({level:.1%})",
        }
        for i, (level, frac) in enumerate(zip(levels, fractions, strict=True))
    ]


def _stops_from_tranche(tranche: TrailingStopTranchePolicy) -> list[dict[str, Any]]:
    return _stops_from_levels_and_fractions(tranche.levels, tranche.fractions)


def _clamp_level(level: float, policy: TrailingStopPolicy) -> float:
    """Clamp drawdown level between policy min (widest) and max (tightest)."""
    return max(policy.min_level, min(policy.max_level, level))


def _sentiment_bucket(
    sentiment_score: float,
    policy: TrailingStopPolicy,
) -> tuple[TrailingStopTranchePolicy, str, float]:
    mults = policy.sentiment_vol_mult
    if sentiment_score >= policy.bull_threshold:
        return policy.bull, "Bullish-derived", float(mults.get("bull", 1.0))
    if sentiment_score <= policy.bear_threshold:
        return policy.bear, "Bearish-derived", float(mults.get("bear", 1.0))
    return policy.neutral, "Neutral-derived", float(mults.get("neutral", 1.0))


def vol_scaled_stop_levels(
    daily_vol: float,
    policy: TrailingStopPolicy,
    *,
    sentiment_mult: float = 1.0,
) -> tuple[tuple[float, float, float], float]:
    """Compute three drawdown levels from daily vol, horizon, sigmas, and sentiment mult."""
    vol = max(float(daily_vol), 1e-8)
    scale = vol * math.sqrt(max(policy.horizon_days, 1)) * sentiment_mult
    levels = tuple(
        _clamp_level(-sigma * scale, policy) for sigma in policy.tranche_sigmas
    )
    return levels, scale


def build_stops(
    sentiment_score: float,
    policy: TrailingStopPolicy | None = None,
    *,
    daily_vol: float | None = None,
    use_manual: bool | None = None,
    manual_levels: tuple[float, ...] | None = None,
    manual_fractions: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], str, float | None, float | None]:
    """Return stop dicts, regime tag, daily vol used, and vol scale (if vol_scaled)."""
    policy = policy or DEFAULT_TRAILING_STOP_POLICY
    manual = use_manual if use_manual is not None else policy.use_manual
    vol_used: float | None = None
    vol_scale: float | None = None

    if manual:
        tranche = policy.manual
        if manual_levels is not None or manual_fractions is not None:
            lv = manual_levels or tranche.levels
            fr = manual_fractions or tranche.fractions
            tranche = TrailingStopTranchePolicy((lv[0], lv[1], lv[2]), (fr[0], fr[1], fr[2]))
        stops, tag = _stops_from_tranche(tranche), "Manual"
    elif policy.mode == "vol_scaled":
        tranche, tag, sentiment_mult = _sentiment_bucket(sentiment_score, policy)
        vol_used = float(daily_vol if daily_vol is not None else policy.fallback_daily_vol)
        levels, vol_scale = vol_scaled_stop_levels(vol_used, policy, sentiment_mult=sentiment_mult)
        stops = _stops_from_levels_and_fractions(levels, tranche.fractions)
        tag = f"{tag} (vol-scaled)"
    else:
        tranche, tag, _ = _sentiment_bucket(sentiment_score, policy)
        stops = _stops_from_tranche(tranche)

    return stops, tag, vol_used, vol_scale


def trailing_stop_set(
    sentiment_score: float,
    policy: TrailingStopPolicy | None = None,
    **kwargs: Any,
) -> TrailingStopSet:
    """Build a TrailingStopSet for one ticker."""
    stops, tag, vol_used, vol_scale = build_stops(sentiment_score, policy, **kwargs)
    return TrailingStopSet(
        sentiment_score=sentiment_score,
        regime_tag=tag,
        stops=(stops[0], stops[1], stops[2]),
        daily_vol=vol_used,
        vol_scale=vol_scale,
    )
