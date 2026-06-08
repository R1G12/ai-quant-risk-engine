"""Convert tracker amounts from quote currency to display currency (per-trade-date FX)."""

from __future__ import annotations

from datetime import date

import polars as pl

from src.portfolio.tracker.fx import FxRateTable
from src.portfolio.tracker.prices import MarkPriceInfo
from src.portfolio.tracker.schema import cash_flow, signed_quantity_delta
from src.utils.config import AppConfig, PortfolioTrackerConfig


def fx_conversion_enabled(cfg: PortfolioTrackerConfig) -> bool:
    return cfg.display_currency.upper() != cfg.quote_currency.upper()


def fx_date_span(
    trades: pl.DataFrame,
    mark_info: dict[str, MarkPriceInfo],
) -> tuple[date, date]:
    """Min/max calendar span for FX history."""
    today = date.today()
    dates: list[date] = [today]
    if trades.height:
        dates.append(trades["trade_date"].min())
        dates.append(trades["trade_date"].max())
    for info in mark_info.values():
        if info.as_of_date is not None:
            dates.append(info.as_of_date)
    return min(dates), max(dates)


def load_fx_table(
    app: AppConfig,
    trades: pl.DataFrame,
    mark_info: dict[str, MarkPriceInfo],
) -> FxRateTable | None:
    cfg = app.tracker
    if not fx_conversion_enabled(cfg):
        return None
    start, end = fx_date_span(trades, mark_info)
    return FxRateTable.load(cfg.fx_pair, start, end)


def realized_pnl_sgd(lot_trades: pl.DataFrame, fx: FxRateTable) -> float:
    """Sum cash flows at each trade's FX rate (closed lot)."""
    total = 0.0
    for row in lot_trades.sort("trade_date", "trade_id").iter_rows(named=True):
        cf = cash_flow(
            row["side"],
            row["action"],
            row["quantity"],
            row["price"],
            row["fees"],
        )
        total += cf * fx.rate_on(row["trade_date"])
    return total


def _cost_basis_sgd_long(lot: pl.DataFrame, fx: FxRateTable) -> float:
    qty = 0.0
    cost = 0.0
    for row in lot.sort("trade_date", "trade_id").iter_rows(named=True):
        delta = signed_quantity_delta(row["side"], row["action"], row["quantity"])
        r = fx.rate_on(row["trade_date"])
        if delta > 0:
            cost += row["quantity"] * row["price"] * r + row["fees"] * r
        elif delta < 0 and qty > 1e-12:
            cost *= (qty + delta) / qty
        qty += delta
    return cost


def _liability_sgd_short(lot: pl.DataFrame, fx: FxRateTable) -> float:
    qty = 0.0
    liability = 0.0
    for row in lot.sort("trade_date", "trade_id").iter_rows(named=True):
        delta = signed_quantity_delta(row["side"], row["action"], row["quantity"])
        r = fx.rate_on(row["trade_date"])
        if delta > 0:
            liability += row["quantity"] * row["price"] * r - row["fees"] * r
        elif delta < 0 and qty > 1e-12:
            liability *= (qty + delta) / qty
        qty += delta
    return liability


def open_lot_sgd(
    lot_trades: pl.DataFrame,
    *,
    quantity: float,
    side: str,
    mark: float,
    mark_date: date,
    fx: FxRateTable,
) -> tuple[float, float]:
    """Return (market_value_sgd, unrealized_pnl_sgd) for an open lot."""
    mark_r = fx.rate_on(mark_date)
    market_value = quantity * mark * mark_r
    if side == "short":
        liability = _liability_sgd_short(lot_trades, fx)
        return market_value, liability - market_value
    cost = _cost_basis_sgd_long(lot_trades, fx)
    return market_value, market_value - cost


def apply_display_currency_to_open(
    open_pos: pl.DataFrame,
    trades: pl.DataFrame,
    mark_info: dict[str, MarkPriceInfo],
    fx: FxRateTable,
) -> pl.DataFrame:
    if open_pos.is_empty():
        return open_pos
    rows = []
    for row in open_pos.iter_rows(named=True):
        lot_id = row["lot_id"]
        ticker = row["ticker"]
        lot = trades.filter(pl.col("lot_id") == lot_id)
        info = mark_info.get(ticker)
        if info is None or info.price is None or info.as_of_date is None:
            rows.append({**row, "market_value": None, "unrealized_pnl": None})
            continue
        mv, ur = open_lot_sgd(
            lot,
            quantity=float(row["quantity"]),
            side=row["side"],
            mark=info.price,
            mark_date=info.as_of_date,
            fx=fx,
        )
        rows.append({**row, "market_value": mv, "unrealized_pnl": ur})
    return pl.DataFrame(rows)


def apply_display_currency_to_closed(
    closed_pos: pl.DataFrame,
    trades: pl.DataFrame,
    fx: FxRateTable,
) -> pl.DataFrame:
    if closed_pos.is_empty():
        return closed_pos
    rows = []
    for row in closed_pos.iter_rows(named=True):
        lot = trades.filter(pl.col("lot_id") == row["lot_id"])
        pnl = realized_pnl_sgd(lot, fx)
        fee_sgd = 0.0
        for trow in lot.iter_rows(named=True):
            fee_sgd += trow["fees"] * fx.rate_on(trow["trade_date"])
        rows.append({**row, "realized_pnl": pnl, "total_fees": fee_sgd})
    return pl.DataFrame(rows)


def exposure_notional_sgd(
    open_pos: pl.DataFrame,
    mark_info: dict[str, MarkPriceInfo],
    fx: FxRateTable,
) -> pl.DataFrame:
    """Signed notional in display currency for vs-model comparison."""
    if open_pos.is_empty():
        return pl.DataFrame({"ticker": [], "notional": [], "weight": []})

    rows = []
    for row in open_pos.iter_rows(named=True):
        ticker = row["ticker"]
        info = mark_info.get(ticker)
        if info is None or info.price is None or info.as_of_date is None:
            continue
        r = fx.rate_on(info.as_of_date)
        sign = 1.0 if row["side"] == "long" else -1.0
        notional = sign * float(row["quantity"]) * info.price * r
        rows.append({"ticker": ticker, "notional": notional})

    if not rows:
        return pl.DataFrame({"ticker": [], "notional": [], "weight": []})

    df = pl.DataFrame(rows).group_by("ticker").agg(pl.col("notional").sum().alias("notional"))
    gross = float(df["notional"].abs().sum())
    if gross < 1e-12:
        return df.with_columns(pl.lit(0.0).alias("weight"))
    return df.with_columns((pl.col("notional").abs() / gross).alias("weight"))


def currency_context(app: AppConfig, fx: FxRateTable | None) -> dict[str, str | None]:
    cfg = app.tracker
    ctx: dict[str, str | None] = {
        "quote": cfg.quote_currency,
        "display": cfg.display_currency,
        "fx_pair": cfg.fx_pair if fx_conversion_enabled(cfg) else None,
        "fx_latest": None,
    }
    if fx is not None and fx.latest_date() is not None:
        ctx["fx_latest"] = str(fx.latest_date())
    return ctx
