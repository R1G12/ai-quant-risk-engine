"""Helpers for trading_risk_manager_final.ipynb (gross budget, partial opt, per-ticker stops)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

Side = Literal["long", "short"]


def side_for_ticker(ticker: str, position_sides: dict[str, str] | None) -> Side:
    if position_sides and ticker in position_sides:
        s = position_sides[ticker].strip().lower()
        if s not in ("long", "short"):
            raise ValueError(f"POSITION_SIDES[{ticker!r}] must be 'long' or 'short', got {s!r}")
        return s  # type: ignore[return-value]
    return "long"


def sign_for_ticker(ticker: str, position_sides: dict[str, str] | None) -> float:
    return -1.0 if side_for_ticker(ticker, position_sides) == "short" else 1.0


def normalize_gross_weights(
    weights: np.ndarray,
    *,
    allow_shorts: bool = True,
    max_gross_per_ticker: float = 1.0,
) -> np.ndarray:
    """Scale so sum(abs(w)) == 1; clip to per-ticker gross cap."""
    w = np.asarray(weights, dtype=float).copy()
    if not allow_shorts:
        w = np.clip(w, 0.0, max_gross_per_ticker)
    else:
        w = np.clip(w, -max_gross_per_ticker, max_gross_per_ticker)
    gross = float(np.sum(np.abs(w)))
    if gross < 1e-12:
        raise ValueError("Weights are all zero; cannot normalize gross budget.")
    return w / gross


def apply_position_sides(
    weights: np.ndarray,
    tickers: list[str],
    position_sides: dict[str, str] | None,
) -> np.ndarray:
    """Force sign from POSITION_SIDES while preserving magnitudes."""
    w = np.asarray(weights, dtype=float).copy()
    for i, t in enumerate(tickers):
        w[i] = abs(w[i]) * sign_for_ticker(t, position_sides)
    return w


def exposure_summary(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, dtype=float)
    return {
        "gross": float(np.sum(np.abs(w))),
        "net": float(np.sum(w)),
        "long": float(np.sum(w[w > 0])),
        "short": float(np.sum(w[w < 0])),
    }


def _bounds(n: int, *, allow_shorts: bool, max_gross_per_ticker: float) -> list[tuple[float, float]]:
    if allow_shorts:
        return [(-max_gross_per_ticker, max_gross_per_ticker)] * n
    return [(0.0, max_gross_per_ticker)] * n


def optimize_max_sharpe_gross(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    *,
    risk_free: float = 0.0,
    allow_shorts: bool = True,
    max_gross_per_ticker: float = 0.5,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Max Sharpe with sum(|w|) = 1."""
    n = len(mean_returns)
    x0 = x0 if x0 is not None else np.ones(n) / n
    x0 = normalize_gross_weights(x0, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mean_returns)
        vol = float(np.sqrt(w @ cov @ w))
        if vol < 1e-12:
            return 0.0
        return -(ret - risk_free) / vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=_bounds(n, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker),
        constraints=cons,
    )
    w = res.x if res.success else x0
    return w, bool(res.success)


def optimize_partial_weights(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    anchor_weights: dict[str, float],
    *,
    risk_free: float = 0.0,
    allow_shorts: bool = True,
    max_gross_per_ticker: float = 0.5,
    position_sides: dict[str, str] | None = None,
) -> tuple[np.ndarray, bool]:
    """Fix anchor tickers; optimize free tickers with gross budget on remaining sleeve."""
    n = len(tickers)
    w = np.zeros(n)
    fixed_idx: list[int] = []
    for t, val in anchor_weights.items():
        if t not in tickers:
            raise ValueError(f"ANCHOR_WEIGHTS unknown ticker: {t!r}")
        i = tickers.index(t)
        w[i] = float(val)
        fixed_idx.append(i)

    for t, val in anchor_weights.items():
        i = tickers.index(t)
        w[i] = float(val)
        if position_sides and t in position_sides:
            w[i] = abs(w[i]) * sign_for_ticker(t, position_sides)

    gross_anchor = float(np.sum(np.abs(w)))
    free_idx = [i for i in range(n) if i not in fixed_idx]
    target_free_gross = 1.0 - gross_anchor

    if target_free_gross <= 1e-9:
        if free_idx:
            free_names = [tickers[i] for i in free_idx]
            raise ValueError(
                f"ANCHOR_WEIGHTS use {gross_anchor:.1%} gross budget — no room left for "
                f"free tickers {free_names}. Reduce anchor magnitudes so sum(|anchors|) < 1."
            )
        return normalize_gross_weights(
            w, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker
        ), True

    if not free_idx:
        return normalize_gross_weights(
            w, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker
        ), True

    def pack(free_w: np.ndarray) -> np.ndarray:
        full = w.copy()
        for j, i in enumerate(free_idx):
            full[i] = free_w[j]
        return full

    def neg_sharpe(free_w: np.ndarray) -> float:
        full = pack(free_w)
        ret = float(full @ mean_returns)
        vol = float(np.sqrt(full @ cov @ full))
        if vol < 1e-12:
            return 0.0
        return -(ret - risk_free) / vol

    def gross_constraint(free_w: np.ndarray) -> float:
        full = pack(free_w)
        return float(np.sum(np.abs(full)) - 1.0)

    x0_free = np.full(len(free_idx), target_free_gross / len(free_idx))
    cons = [{"type": "eq", "fun": gross_constraint}]
    res = minimize(
        neg_sharpe,
        x0_free,
        method="SLSQP",
        bounds=_bounds(len(free_idx), allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker),
        constraints=cons,
    )
    w_opt = pack(res.x if res.success else x0_free)
    w_opt = apply_position_sides(w_opt, tickers, position_sides)
    return w_opt, bool(res.success)


def resolve_weights(
    mode: str,
    tickers: list[str],
    *,
    allow_shorts: bool,
    max_gross_per_ticker: float,
    position_sides: dict[str, str] | None,
    manual_weights: dict[str, float] | None = None,
    anchor_weights: dict[str, float] | None = None,
    w_optimised: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    risk_free: float = 0.0,
) -> np.ndarray:
    """Resolve weight vector for equal | manual | optimised | partial."""
    n = len(tickers)
    if mode == "equal":
        signs = np.array([sign_for_ticker(t, position_sides) for t in tickers])
        w = signs / n
        return normalize_gross_weights(w, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker)

    if mode == "manual":
        if not manual_weights:
            raise ValueError("MANUAL_WEIGHTS required for mode='manual'")
        w = np.array([float(manual_weights.get(t, 0.0)) for t in tickers])
        if not allow_shorts and np.any(w < 0):
            raise ValueError("Negative manual weight but ALLOW_SHORTS is False")
        gross = float(np.sum(np.abs(w)))
        if abs(gross - 1.0) > 1e-3:
            w = normalize_gross_weights(w, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker)
        return w

    if mode == "optimised":
        if w_optimised is None:
            raise NameError("w_sharpe")
        w = np.asarray(w_optimised, dtype=float)
        return apply_position_sides(
            normalize_gross_weights(w, allow_shorts=allow_shorts, max_gross_per_ticker=max_gross_per_ticker),
            tickers,
            position_sides,
        )

    if mode == "partial":
        if not anchor_weights:
            raise ValueError("ANCHOR_WEIGHTS required for mode='partial'")
        if w_optimised is not None and mean_returns is None:
            return apply_position_sides(
                np.asarray(w_optimised, dtype=float), tickers, position_sides
            )
        if mean_returns is None or cov is None:
            raise NameError("w_sharpe")
        w, _ = optimize_partial_weights(
            mean_returns,
            cov,
            tickers,
            anchor_weights,
            risk_free=risk_free,
            allow_shorts=allow_shorts,
            max_gross_per_ticker=max_gross_per_ticker,
            position_sides=position_sides,
        )
        return w

    raise ValueError(f"Unknown PORTFOLIO_WEIGHTING: {mode!r}")


def simulate_trailing_stops(
    paths: np.ndarray,
    stops: list[dict[str, Any]],
    side: Side = "long",
) -> np.ndarray:
    """Apply 3-tranche trailing stops on position MTM paths (HORIZON+1, N_PATHS).

    For shorts, pass position-value paths (e.g. GBM with negated returns), not raw
    underlying prices — then use side='long'. The ``side='short'`` branch mirrors
  peak/trough logic when paths track underlying price instead of MTM.
    """
    horizon, n_paths = paths.shape[0] - 1, paths.shape[1]
    stops_sorted = sorted(stops, key=lambda s: s["level"])
    out = np.zeros_like(paths)
    out[0] = paths[0]

    for p in range(n_paths):
        in_mkt = float(paths[0, p])
        cash = 0.0
        peak = in_mkt
        trough = in_mkt
        triggered = [False] * len(stops_sorted)

        for day in range(1, horizon + 1):
            prev = paths[day - 1, p]
            in_mkt *= paths[day, p] / prev if prev != 0 else 1.0
            if side == "long":
                peak = max(peak, in_mkt)
                dd = (in_mkt - peak) / peak if peak > 0 else 0.0
                breach = lambda dd, lvl: dd <= lvl
            else:
                trough = min(trough, in_mkt) if in_mkt > 0 else trough
                dd = (in_mkt - trough) / trough if trough > 0 else 0.0
                breach = lambda dd, lvl: dd >= abs(lvl)

            for i, stop in enumerate(stops_sorted):
                if not triggered[i] and breach(dd, stop["level"]):
                    locked = stop["exit_fraction"] * in_mkt
                    cash += locked
                    in_mkt -= locked
                    triggered[i] = True
            out[day, p] = in_mkt + cash

    return out


def aggregate_portfolio_paths(
    path_by_ticker: dict[str, np.ndarray],
    weights: np.ndarray,
    tickers: list[str],
) -> np.ndarray:
    """Gross-weighted sum of per-ticker stopped paths."""
    total = None
    w = np.asarray(weights, dtype=float)
    for i, t in enumerate(tickers):
        contrib = np.abs(w[i]) * path_by_ticker[t]
        total = contrib if total is None else total + contrib
    return total if total is not None else np.array([])


def gbm_paths(
    *,
    horizon: int,
    n_paths: int,
    start_value: float,
    drift_daily: float,
    vol_daily: float,
    seed: int | None = None,
    side: Side = "long",
) -> np.ndarray:
    """Raw GBM position MTM paths without stops, shape (horizon+1, n_paths)."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1, (horizon, n_paths))
    sign = -1.0 if side == "short" else 1.0
    log_ret = sign * ((drift_daily - 0.5 * vol_daily**2) + vol_daily * shocks)
    return np.vstack([np.full(n_paths, start_value), start_value * np.exp(np.cumsum(log_ret, axis=0))])
