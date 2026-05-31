"""Backtest performance metrics."""

from __future__ import annotations

import polars as pl


def sharpe_ratio(returns: pl.Series, rf: float, ann: int = 252) -> float:
    r = returns.drop_nulls()
    if r.len() < 2:
        return 0.0
    excess = float(r.mean()) * ann - rf
    vol = float(r.std()) * (ann**0.5)
    return excess / vol if vol > 1e-12 else 0.0


def max_drawdown(equity: pl.Series) -> float:
    dd = equity / equity.cum_max() - 1.0
    return float(dd.min())
