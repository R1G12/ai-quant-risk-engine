"""Transaction costs and slippage."""

from __future__ import annotations


def trading_cost(turnover: float, tc_bps: float, slippage_bps: float) -> float:
    """Total cost as fraction of portfolio value from turnover."""
    bps = tc_bps + slippage_bps
    return turnover * bps / 10_000.0
