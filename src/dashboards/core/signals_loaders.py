"""Load FinBERT scores, regimes, and stop tables for the Signals dashboard page."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

from src.portfolio.sentiment_sides import load_finbert_window as _load_finbert_window
from src.portfolio.sentiment_sides import map_sentiment_dataframe
from src.portfolio.stops import TrailingStopSet, resolve_trailing_stop_policy, trailing_stop_set
from src.risk.portfolio.holdings import load_weights
from src.utils.config import AppConfig
from src.utils import paths as project_paths
from src.utils.paths import PROCESSED_SENTIMENT_PATH, RISK_REGIMES_PATH

# Gaussian HMM labels from src.risk.regimes.hmm (mean-return ordering).
HMM_REGIME_LABELS: tuple[str, ...] = ("low", "mid", "high")
REGIME_HISTORY_OBS = 365
RECENT_SCORE_HISTORY_DAYS = 3


def load_finbert_window(
    app: AppConfig,
    *,
    window_days: int = 30,
    tickers: list[str] | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Delegate to shared portfolio sentiment loader."""
    return _load_finbert_window(app, window_days=window_days, tickers=tickers)


def _to_day(ts: object) -> date:
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    return date.fromisoformat(str(ts)[:10])


def _utc_timestamp(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def finbert_recent_days_panel(
    daily: pl.DataFrame,
    tickers: list[str],
    *,
    n_days: int = RECENT_SCORE_HISTORY_DAYS,
    window_end: date | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Last ``n_days`` calendar days per ticker on a shared axis, forward-filling stale scores.

    Returns ``panel`` (one row per ticker-day) and ``stale_meta`` (per-ticker disclaimer inputs).
    """
    if daily.is_empty() or not tickers:
        return pl.DataFrame(), pl.DataFrame()

    if window_end is None:
        window_end = _to_day(daily["timestamp"].max())

    day_list = [window_end - timedelta(days=n_days - 1 - i) for i in range(n_days)]

    hist = daily.with_columns(
        pl.col("timestamp")
        .cast(pl.Datetime(time_unit="us", time_zone="UTC"), strict=False)
        .dt.date()
        .alias("day")
    )
    panel_rows: list[dict[str, object]] = []
    stale_rows: list[dict[str, object]] = []

    for ticker in tickers:
        t_hist = hist.filter(pl.col("ticker") == ticker).sort("day")
        if t_hist.is_empty():
            stale_rows.append(
                {
                    "ticker": ticker,
                    "has_stale": False,
                    "no_data": True,
                    "last_article_day": None,
                    "stale_day_labels": [],
                }
            )
            continue

        last_article_day = _to_day(t_hist["day"][-1])
        stale_labels: list[str] = []

        for d in day_list:
            on_day = t_hist.filter(pl.col("day") == d)
            if on_day.height:
                row = on_day.row(0, named=True)
                panel_rows.append(
                    {
                        "ticker": ticker,
                        "day": d,
                        "timestamp": _utc_timestamp(d),
                        "sentiment_score": float(row["sentiment_score"]),
                        "is_forward_filled": False,
                        "score_as_of": d,
                    }
                )
                continue

            prior = t_hist.filter(pl.col("day") <= d)
            if prior.height:
                carry = prior.tail(1).row(0, named=True)
                as_of = _to_day(carry["day"])
            else:
                after = t_hist.filter(pl.col("day") > d).sort("day").head(1)
                if after.is_empty():
                    continue
                carry = after.row(0, named=True)
                as_of = _to_day(carry["day"])
            panel_rows.append(
                {
                    "ticker": ticker,
                    "day": d,
                    "timestamp": _utc_timestamp(d),
                    "sentiment_score": float(carry["sentiment_score"]),
                    "is_forward_filled": True,
                    "score_as_of": as_of,
                }
            )
            stale_labels.append(d.isoformat())

        stale_rows.append(
            {
                "ticker": ticker,
                "has_stale": bool(stale_labels),
                "no_data": False,
                "last_article_day": last_article_day,
                "stale_day_labels": stale_labels,
            }
        )

    panel = pl.DataFrame(panel_rows) if panel_rows else pl.DataFrame()
    stale_meta = pl.DataFrame(stale_rows) if stale_rows else pl.DataFrame()
    if panel.height:
        panel = panel.sort(["ticker", "day"]).with_columns(
            pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        )
    return panel, stale_meta


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


def load_latest_daily_vol_by_ticker(tickers: list[str] | None = None) -> dict[str, float]:
    """Latest rolling daily vol (returns std) per ticker from the feature store."""
    path = project_paths.FEATURES_VOLATILITY_DIR / "volatility.parquet"
    if not path.is_file():
        return {}
    want = {str(t).upper() for t in tickers} if tickers else None
    df = (
        pl.scan_parquet(path)
        .filter(pl.col("volatility").is_not_null())
        .sort(["ticker", "timestamp"])
        .group_by("ticker")
        .agg(pl.col("volatility").last().alias("daily_vol"))
        .collect()
    )
    out: dict[str, float] = {}
    for row in df.iter_rows(named=True):
        ticker = str(row["ticker"]).upper()
        if want is not None and ticker not in want:
            continue
        out[ticker] = float(row["daily_vol"])
    return out


def load_trailing_stops_table(summary: pl.DataFrame, app: AppConfig | None = None) -> pl.DataFrame:
    """Expand sentiment summary into per-ticker stop levels (3 tranches)."""
    policy = resolve_trailing_stop_policy(app)
    tickers = [str(t) for t in summary["ticker"].to_list()]
    vol_by_ticker = (
        load_latest_daily_vol_by_ticker(tickers) if policy.mode == "vol_scaled" else {}
    )
    rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        ticker = str(row["ticker"])
        score = float(row["sentiment_score"])
        daily_vol = vol_by_ticker.get(ticker.upper())
        tset: TrailingStopSet = trailing_stop_set(score, policy, daily_vol=daily_vol)
        rows.append(
            {
                "ticker": ticker,
                "sentiment_score": score,
                "regime_tag": tset.regime_tag,
                "daily_vol": tset.daily_vol,
                "vol_scale": tset.vol_scale,
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
