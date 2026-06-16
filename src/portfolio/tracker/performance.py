"""Model vs actual portfolio performance curves for Tracker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from src.analytics.data_bounds import detect_data_bounds
from src.features.wide_returns import load_returns_wide
from src.market.adapters.yfinance import download_symbol_history, yfinance_close_series, yfinance_scalar_float
from src.portfolio.tracker.fx import FxRateTable
from src.portfolio.tracker.schema import cash_flow, economic_position_delta
from src.risk.portfolio.weights_loader import load_optimization_weights
from src.utils.config import AppConfig


@dataclass(frozen=True)
class ComparisonBounds:
    min_date: date
    max_date: date


def tracker_comparison_bounds(trades: pl.DataFrame, app: AppConfig) -> ComparisonBounds:
    """Date window for Tracker comparison UI.

    Extends through yesterday and the last trade date. Unlike the main pipeline
    slider, does not cap *down* to stale processed-market rows for book tickers.
    """
    pipeline = detect_data_bounds(app.research.meta.experiment_id)
    yesterday = date.today() - timedelta(days=1)

    min_d = pipeline.min_date
    max_d = max(pipeline.max_date, yesterday)

    if trades.height:
        first = trades["trade_date"].min()
        last = trades["trade_date"].max()
        if first is not None:
            min_d = max(min_d, first)
        if last is not None:
            max_d = max(max_d, last, yesterday)

    max_d = min(max_d, yesterday)

    if min_d > max_d:
        max_d = min_d
    return ComparisonBounds(min_date=min_d, max_date=max_d)


def _weighted_portfolio_returns(wide: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    port = pl.lit(0.0)
    for ticker, w in weights.items():
        if ticker in wide.columns:
            port = port + pl.col(ticker).fill_null(0.0) * w
    return (
        wide.with_columns(port.alias("daily_return"))
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .select("date", "daily_return")
        .sort("date")
    )


def _indexed_equity_from_returns(rets: pl.DataFrame) -> pl.DataFrame:
    """Cumulative product of daily_return, indexed to 100 at first row."""
    if rets.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64, "equity_indexed": pl.Float64})
    rets = rets.with_columns((1.0 + pl.col("daily_return")).cum_prod().alias("growth"))
    base = float(rets["growth"][0])
    if abs(base) < 1e-12:
        base = 1.0
    return rets.with_columns((100.0 * pl.col("growth") / base).alias("equity_indexed"))


def _benchmark_returns_from_yfinance(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Daily returns for benchmark ticker when absent from risk_dataset."""
    buffer_start = start - timedelta(days=7)
    hist = download_symbol_history(ticker, buffer_start, end)
    series = yfinance_close_series(hist)
    if series is None or series.empty:
        return pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64})

    closes = [yfinance_scalar_float(v) for v in series.tolist()]
    dates = [d.date() if hasattr(d, "date") else d for d in series.index]
    df = pl.DataFrame({"date": dates, "close": closes}).sort("date")
    df = df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("daily_return"))
    return df.select("date", "daily_return").filter(pl.col("date") >= start, pl.col("date") <= end)


def benchmark_equity_curve(app: AppConfig, start: date, end: date) -> pl.DataFrame:
    """Cumulative buy-and-hold benchmark (e.g. SPY), indexed to 100 at range start."""
    empty = pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64, "equity_indexed": pl.Float64})
    ticker = app.tracker.benchmark_ticker
    if not ticker:
        return empty

    wide = load_returns_wide([ticker])
    if wide.is_empty() or ticker not in wide.columns:
        rets = _benchmark_returns_from_yfinance(ticker, start, end)
        return _indexed_equity_from_returns(rets)

    rets = (
        wide.with_columns(pl.col(ticker).fill_null(0.0).alias("daily_return"))
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .select("date", "daily_return")
        .sort("date")
        .filter(pl.col("date") >= start, pl.col("date") <= end)
    )
    if rets.is_empty():
        rets = _benchmark_returns_from_yfinance(ticker, start, end)
    return _indexed_equity_from_returns(rets)


def model_equity_curve(app: AppConfig, start: date, end: date) -> pl.DataFrame:
    """Cumulative model equity from optimized weights and risk_dataset returns."""
    weights, _ = load_optimization_weights(app)
    if not weights:
        return pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64, "equity": pl.Float64})

    wide = load_returns_wide(list(weights.keys()))
    if wide.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64, "equity": pl.Float64})

    rets = _weighted_portfolio_returns(wide, weights)
    rets = rets.filter(pl.col("date") >= start, pl.col("date") <= end)
    if rets.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "daily_return": pl.Float64, "equity": pl.Float64})

    rets = rets.with_columns((1.0 + pl.col("daily_return")).cum_prod().alias("growth"))
    base = float(rets["growth"][0])
    if abs(base) < 1e-12:
        base = 1.0
    return rets.with_columns((100.0 * pl.col("growth") / base).alias("equity"))


def _close_on_date(panel: pl.DataFrame, ticker: str, d: date, last: dict[str, float]) -> float | None:
    row = panel.filter((pl.col("ticker") == ticker) & (pl.col("date") == d))
    if row.height:
        px = float(row["close"][0])
        last[ticker] = px
        return px
    return last.get(ticker)


def actual_equity_curve(
    trades: pl.DataFrame,
    panel: pl.DataFrame,
    initial_nav: float,
    start: date,
    end: date,
    *,
    fx: FxRateTable | None = None,
) -> tuple[pl.DataFrame, date | None]:
    """Daily NAV from trade ledger and close panel. Returns (curve, first_trade_date in range).

    When ``fx`` is set, trade cash flows use each trade's FX rate and daily MTM uses
    that day's rate; ``initial_nav`` is in display currency (e.g. SGD).
    """
    empty = pl.DataFrame(schema={"date": pl.Date, "equity": pl.Float64, "equity_indexed": pl.Float64})

    if trades.is_empty():
        return empty, None

    ordered = trades.sort("trade_date", "trade_id")
    first_trade = ordered["trade_date"][0]
    cash = float(initial_nav)
    positions: dict[str, float] = {}
    last_close: dict[str, float] = {}

    trade_by_date: dict[date, list[dict]] = {}
    for row in ordered.iter_rows(named=True):
        d = row["trade_date"]
        trade_by_date.setdefault(d, []).append(row)

    curve_start = max(start, first_trade)
    if curve_start > end:
        return empty, first_trade

    dates: list[date] = []
    equities: list[float] = []
    d = curve_start
    while d <= end:
        for row in trade_by_date.get(d, []):
            delta = economic_position_delta(row["side"], row["action"], row["quantity"])
            ticker = row["ticker"]
            positions[ticker] = positions.get(ticker, 0.0) + delta
            if abs(positions[ticker]) < 1e-9:
                positions.pop(ticker, None)
            cf_usd = cash_flow(
                row["side"], row["action"], row["quantity"], row["price"], row["fees"]
            )
            if fx is not None:
                cash += cf_usd * fx.rate_on(row["trade_date"])
            else:
                cash += cf_usd

        mtm = 0.0
        fx_d = fx.rate_on(d) if fx is not None else 1.0
        for ticker, qty in positions.items():
            px = _close_on_date(panel, ticker, d, last_close)
            if px is not None:
                mtm += qty * px * fx_d

        dates.append(d)
        equities.append(cash + mtm)
        d += timedelta(days=1)

    if not dates:
        return empty, first_trade

    df = pl.DataFrame({"date": dates, "equity": equities})
    base = float(df["equity"][0])
    if abs(base) < 1e-12:
        base = 1.0
    df = df.with_columns((100.0 * pl.col("equity") / base).alias("equity_indexed"))
    return df, first_trade


def preset_range(preset: str, bounds: ComparisonBounds) -> tuple[date, date]:
    """Map 3M/6M/1Y/All to (start, end)."""
    end = bounds.max_date
    if preset == "All":
        return bounds.min_date, end
    days = {"3M": 90, "6M": 180, "1Y": 365}.get(preset, 365)
    start = max(bounds.min_date, end - timedelta(days=days))
    return start, end
