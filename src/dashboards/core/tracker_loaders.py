"""Load trade ledger and derived positions for Tracker dashboard."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from src.analytics.charts.context import DateRange
from src.portfolio.tracker.ingest import ingest_trades, resolve_excel_path
from src.portfolio.tracker.performance import (
    ComparisonBounds,
    actual_equity_curve,
    model_equity_curve,
    tracker_comparison_bounds,
)
from src.portfolio.tracker.currency import (
    apply_display_currency_to_closed,
    apply_display_currency_to_open,
    currency_context,
    exposure_notional_sgd,
    load_fx_table,
)
from src.portfolio.tracker.pnl import attach_unrealized_pnl, exposure_by_ticker
from src.portfolio.tracker.positions import build_closed_positions, build_open_positions
from src.portfolio.tracker.prices import close_panel, latest_mark_prices_with_info
from src.utils.config import AppConfig
from src.utils.paths import PROJECT_ROOT


def _tracker_parquet_path(app: AppConfig) -> Path:
    path = Path(app.tracker.trades_parquet)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _needs_ingest(app: AppConfig, parquet_path: Path) -> bool:
    if not parquet_path.is_file():
        return True
    try:
        excel_path, _ = resolve_excel_path(app)
    except FileNotFoundError:
        return False
    return excel_path.stat().st_mtime > parquet_path.stat().st_mtime


def load_tracker_metadata(app: AppConfig) -> dict | None:
    meta_path = Path(app.tracker.metadata_json)
    if not meta_path.is_absolute():
        meta_path = PROJECT_ROOT / meta_path
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_tracker_bundle(app: AppConfig, *, refresh: bool = True) -> dict:
    """Return trades, open, closed, marks, mark_info, exposure, metadata."""
    parquet_path = _tracker_parquet_path(app)
    if refresh and _needs_ingest(app, parquet_path):
        ingest_trades(app, force=True)
    elif refresh and not parquet_path.is_file():
        ingest_trades(app, force=True)

    if not parquet_path.is_file():
        raise FileNotFoundError(
            "No trades.parquet. Run `aqre tracker ingest` or add Excel under data/input/portfolio/."
        )

    trades = pl.read_parquet(parquet_path)
    open_pos = build_open_positions(trades)
    closed_pos = build_closed_positions(trades)
    tickers = trades["ticker"].unique().to_list()
    mark_info = latest_mark_prices_with_info(tickers, app)
    marks = {t: m.price for t, m in mark_info.items() if m.price is not None}
    open_pos = attach_unrealized_pnl(open_pos, marks)
    exposure = exposure_by_ticker(open_pos, marks)

    fx = load_fx_table(app, trades, mark_info)
    if fx is not None:
        open_pos = apply_display_currency_to_open(open_pos, trades, mark_info, fx)
        closed_pos = apply_display_currency_to_closed(closed_pos, trades, fx)
        exposure = exposure_notional_sgd(open_pos, mark_info, fx)

    return {
        "trades": trades,
        "open": open_pos,
        "closed": closed_pos,
        "marks": marks,
        "mark_info": mark_info,
        "exposure": exposure,
        "metadata": load_tracker_metadata(app),
        "currency": currency_context(app, fx),
    }


def load_comparison_bounds(app: AppConfig, trades: pl.DataFrame) -> ComparisonBounds:
    return tracker_comparison_bounds(trades, app)


def load_comparison_curves(
    app: AppConfig,
    trades: pl.DataFrame,
    dr: DateRange,
    initial_nav: float,
) -> dict:
    """Model and actual equity curves for the vs model tab."""
    panel = close_panel(
        trades["ticker"].unique().to_list() if trades.height else [],
        dr.start,
        dr.end,
    )
    model = model_equity_curve(app, dr.start, dr.end)
    if model.height and "equity_indexed" not in model.columns:
        model = model.rename({"equity": "equity_indexed"})

    mark_info = latest_mark_prices_with_info(
        trades["ticker"].unique().to_list() if trades.height else [],
        app,
    )
    fx = load_fx_table(app, trades, mark_info)
    actual, first_trade = actual_equity_curve(
        trades, panel, initial_nav, dr.start, dr.end, fx=fx,
    )
    return {
        "model": model,
        "actual": actual,
        "first_trade_date": first_trade,
        "panel": panel,
        "currency": currency_context(app, fx),
    }
