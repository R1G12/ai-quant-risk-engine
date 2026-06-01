"""Load FinBERT scores, regimes, and stop tables for the Signals dashboard page."""

from __future__ import annotations

import polars as pl

from src.portfolio.sentiment_sides import load_finbert_window
from src.portfolio.stops import TrailingStopSet, trailing_stop_set
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils.paths import RISK_REGIMES_PATH

# Gaussian HMM labels from src.risk.regimes.hmm (mean-return ordering).
HMM_REGIME_LABELS: tuple[str, ...] = ("low", "mid", "high")
REGIME_HISTORY_OBS = 365


def load_finbert_sentiment(
    app: AppConfig,
    *,
    window_days: int = 30,
    tickers: list[str] | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load per-ticker summary and daily scores (defaults to portfolio tickers)."""
    universe = tickers if tickers is not None else list(load_weights(app).keys())
    return load_finbert_window(app, window_days=window_days, tickers=universe)


def load_trailing_stops_table(summary: pl.DataFrame) -> pl.DataFrame:
    """Expand sentiment summary into per-ticker stop levels (3 tranches)."""
    rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        ticker = str(row["ticker"])
        score = float(row["sentiment_score"])
        tset: TrailingStopSet = trailing_stop_set(score)
        rows.append(
            {
                "ticker": ticker,
                "sentiment_score": score,
                "regime_tag": tset.regime_tag,
                "tranche_1_level": tset.tranche_1_level,
                "stop_1_level": tset.stop_1["level"],
                "stop_1_exit": tset.stop_1["exit_fraction"],
                "stop_2_level": tset.stop_2["level"],
                "stop_2_exit": tset.stop_2["exit_fraction"],
                "stop_3_level": tset.stop_3["level"],
                "stop_3_exit": tset.stop_3["exit_fraction"],
            }
        )
    return pl.DataFrame(rows)


def load_latest_regime() -> tuple[str | None, pl.DataFrame | None]:
    """Return (latest_regime_label, full regimes frame) or (None, None)."""
    if not RISK_REGIMES_PATH.is_file():
        return None, None
    regimes = pl.read_parquet(RISK_REGIMES_PATH).sort("timestamp")
    if regimes.is_empty() or "regime_label" not in regimes.columns:
        return None, regimes
    latest = regimes["regime_label"][-1]
    return str(latest) if latest is not None else None, regimes


def regime_history_window(
    regimes_df: pl.DataFrame,
    *,
    n_obs: int = REGIME_HISTORY_OBS,
) -> pl.DataFrame:
    """Last ``n_obs`` regime rows for dashboard history charts."""
    return regimes_df.tail(n_obs)


def regimes_missing_in_window(window: pl.DataFrame) -> list[str]:
    """HMM labels with no rows in ``window`` (for UI hints)."""
    if window.is_empty() or "regime_label" not in window.columns:
        return list(HMM_REGIME_LABELS)
    present = {str(x) for x in window["regime_label"].unique().to_list()}
    return [label for label in HMM_REGIME_LABELS if label not in present]
