"""Load FinBERT scores, regimes, and stop tables for the Signals dashboard page."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.portfolio.sentiment_sides import load_finbert_window as _load_finbert_window
from src.portfolio.sentiment_sides import map_sentiment_dataframe
from src.portfolio.stops import TrailingStopSet, trailing_stop_set
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils.paths import PROCESSED_SENTIMENT_PATH, RISK_REGIMES_PATH

# Gaussian HMM labels from src.risk.regimes.hmm (mean-return ordering).
HMM_REGIME_LABELS: tuple[str, ...] = ("low", "mid", "high")
REGIME_HISTORY_OBS = 365


def load_finbert_window(
    app: AppConfig,
    *,
    window_days: int = 30,
    tickers: list[str] | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Delegate to shared portfolio sentiment loader."""
    return _load_finbert_window(app, window_days=window_days, tickers=tickers)


def finbert_empty_reason(app: AppConfig, *, window_days: int = 30) -> tuple[str, list[str]]:
    """Explain why FinBERT summary is empty (for Signals UI)."""
    if not PROCESSED_SENTIMENT_PATH.is_file():
        return (
            "Sentiment file missing. Run: `dvc repro ingest preprocess sentiment`",
            [],
        )

    raw = pl.read_parquet(PROCESSED_SENTIMENT_PATH)
    if raw.is_empty() or "sentiment_label" not in raw.columns:
        return (
            "Sentiment file is empty or missing FinBERT labels. Re-run the sentiment stage.",
            [],
        )

    mapped = map_sentiment_dataframe(raw, app)
    universe = list(load_weights(app).keys())
    if not universe:
        return ("No holdings tickers in the active run profile.", [])

    max_ts = mapped["timestamp"].max()
    if max_ts is None:
        return ("No timestamps in sentiment data.", universe)

    end = max_ts.date() if hasattr(max_ts, "date") else date.today()
    start = end - timedelta(days=window_days)
    in_window = mapped.filter(
        (pl.col("timestamp").dt.date() >= start) & (pl.col("timestamp").dt.date() <= end)
    )
    in_universe = in_window.filter(pl.col("ticker").is_in(universe))
    if in_universe.height > 0:
        return ("", [])

    tickers_in_data = set(mapped["ticker"].unique().to_list())
    missing = [t for t in universe if t not in tickers_in_data]
    if missing and in_window.height == 0:
        return (
            f"No FinBERT articles in the last {window_days}d for any ticker. "
            "Re-run ingest with `NEWS_SOURCE=yfinance` in live mode.",
            missing,
        )
    return (
        f"No articles in the last {window_days}d for holdings tickers "
        f"({', '.join(universe)}). Mapped sentiment tickers: {sorted(tickers_in_data)}.",
        missing,
    )


def regime_history_window(df: pl.DataFrame, n_obs: int = REGIME_HISTORY_OBS) -> pl.DataFrame:
    """Last ``n_obs`` rows of regime history."""
    if df.is_empty():
        return df
    return df.tail(n_obs)


def regimes_missing_in_window(win: pl.DataFrame) -> list[str]:
    """HMM labels with zero days in the window (axis still shows all three)."""
    if win.is_empty() or "regime_label" not in win.columns:
        return list(HMM_REGIME_LABELS)
    present = set(win["regime_label"].unique().to_list())
    return [lbl for lbl in HMM_REGIME_LABELS if lbl not in present]


def regime_day_counts(win: pl.DataFrame) -> pl.DataFrame:
    """Count trading days per regime label in the window."""
    if win.is_empty() or "regime_label" not in win.columns:
        return pl.DataFrame({"regime_label": list(HMM_REGIME_LABELS), "days": [0, 0, 0]})
    counts = (
        win.group_by("regime_label")
        .agg(pl.len().alias("days"))
        .sort("regime_label")
    )
    rows = []
    count_map = {str(r["regime_label"]): int(r["days"]) for r in counts.iter_rows(named=True)}
    for lbl in HMM_REGIME_LABELS:
        rows.append({"regime_label": lbl, "days": count_map.get(lbl, 0)})
    return pl.DataFrame(rows)


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
