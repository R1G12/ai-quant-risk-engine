"""Rolling correlation features."""

from __future__ import annotations

import polars as pl


def rolling_pairwise_corr_at_date(
    returns_wide: pl.DataFrame,
    tickers: list[str],
    window: int,
) -> pl.DataFrame:
    """Compute correlation matrix at last date using trailing window."""
    sub = returns_wide.tail(window)
    rows = []
    ts = sub["timestamp"][-1] if sub.height else None
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if a not in sub.columns or b not in sub.columns:
                continue
            corr = sub.select(pl.corr(a, b)).item()
            rows.append(
                {
                    "timestamp": ts,
                    "asset_i": a,
                    "asset_j": b,
                    "value": corr,
                    "metric": "corr",
                }
            )
    return pl.DataFrame(rows)
