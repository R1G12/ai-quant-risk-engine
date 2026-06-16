"""Tracker: Excel trade ledger, open/closed positions, charts."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.analytics.charts.context import DateRange
from src.dashboards.core.loaders import load_app
from src.dashboards.core.theme import apply_theme, page_header
from src.dashboards.core.tracker_loaders import (
    load_comparison_bounds,
    load_comparison_curves,
    load_tracker_bundle,
)
from src.portfolio.tracker.formatting import format_money, money_symbol
from src.portfolio.tracker.performance import preset_range
from src.portfolio.tracker.prices import MarkPriceInfo
from src.portfolio.tracker.prices import price_history
from src.risk.portfolio.weights_loader import load_optimization_weights

apply_theme()
app = load_app()
page_header(
    "Tracker",
    "Actual positions from Excel trade ledger (separate from model weights)",
)

try:
    bundle = load_tracker_bundle(app)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Copy `data/input/portfolio/dummy_portfolio.xlsx` to `input_trades.xlsx`, "
        "edit trades, then run `aqre tracker ingest`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load tracker: {exc}")
    st.stop()

mark_info: dict[str, MarkPriceInfo] = bundle.get("mark_info") or {}
_today = date.today()
_stale_sources = [t for t, m in mark_info.items() if m.source == "yfinance"]
_outdated = [
    t
    for t, m in mark_info.items()
    if m.as_of_date is not None and m.as_of_date < _today - timedelta(days=1)
]
_missing = [t for t, m in mark_info.items() if m.source == "missing"]

if _stale_sources or _outdated or _missing:
    parts = [
        "Marks use the **last available daily close** (pipeline parquet or a short yfinance lookback ending yesterday). "
        "Today's bar is not requested — figures can be **slightly outdated**."
    ]
    if _outdated:
        sample = ", ".join(
            f"{t} ({mark_info[t].as_of_date})" for t in _outdated[:5]
        )
        parts.append(f" As-of dates: {sample}.")
    if _missing:
        parts.append(f" No price for: {', '.join(_missing)}.")
    st.warning(" ".join(parts))

meta = bundle.get("metadata") or {}
currency = bundle.get("currency") or {}
display_ccy = str(currency.get("display") or "USD")
quote_ccy = str(currency.get("quote") or "USD")
if meta:
    src = meta.get("source_excel", "")
    st.caption(
        f"Source: `{src}` ({meta.get('source_label', '')}) · "
        f"{meta.get('row_count', 0)} trades · ingested {meta.get('ingested_at', '')}"
    )
if display_ccy != quote_ccy:
    fx_pair = currency.get("fx_pair") or "FX"
    fx_latest = currency.get("fx_latest") or "n/a"
    st.caption(
        f"Amounts in **{display_ccy}**. Trade prices and marks are **{quote_ccy}**; "
        f"P&L converts with **{fx_pair}** on each trade date (marks use mark-date FX). "
        f"Latest FX observation: {fx_latest}."
    )

trades: pl.DataFrame = bundle["trades"]
open_pos: pl.DataFrame = bundle["open"]
closed_pos: pl.DataFrame = bundle["closed"]

tab_open, tab_closed, tab_log, tab_charts, tab_model = st.tabs(
    ["Open", "Closed", "Trade log", "Charts", "vs model"]
)

with tab_open:
    if open_pos.is_empty():
        st.info("No open positions.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Open lots", open_pos.height)
        gross = open_pos.filter(pl.col("market_value").is_not_null())["market_value"].sum()
        c2.metric(
            "Gross market value",
            format_money(float(gross), display_ccy) if gross else "—",
        )
        ur = open_pos.filter(pl.col("unrealized_pnl").is_not_null())["unrealized_pnl"].sum()
        c3.metric(
            "Unrealized P&L",
            format_money(float(ur), display_ccy) if ur is not None else "—",
        )
        st.dataframe(open_pos, width="stretch")

with tab_closed:
    if closed_pos.is_empty():
        st.info("No closed positions yet.")
    else:
        total_realized = float(closed_pos["realized_pnl"].sum())
        st.metric("Total realized P&L", format_money(total_realized, display_ccy, decimals=2))
        st.dataframe(closed_pos, width="stretch")

with tab_log:
    st.dataframe(trades, width="stretch")

with tab_charts:
    tickers = trades["ticker"].unique().sort().to_list()
    if not tickers:
        st.warning("No tickers in trade log.")
    else:
        pick = st.selectbox("Ticker", tickers, key="tracker_chart_ticker")
        ticker_trades = trades.filter(pl.col("ticker") == pick)
        first_trade = ticker_trades["trade_date"].min()
        chart_start = first_trade - timedelta(days=7) if first_trade else None
        hist = price_history(pick, app, start=chart_start)

        fig = go.Figure()
        if hist.height:
            fig.add_trace(
                go.Scatter(
                    x=hist["date"].to_list(),
                    y=hist["close"].to_list(),
                    mode="lines",
                    name="Close",
                )
            )

        buys = ticker_trades.filter(pl.col("action") == "buy")
        sells = ticker_trades.filter(pl.col("action") == "sell")
        if buys.height:
            fig.add_trace(
                go.Scatter(
                    x=buys["trade_date"].to_list(),
                    y=buys["price"].to_list(),
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="lime"),
                )
            )
        if sells.height:
            fig.add_trace(
                go.Scatter(
                    x=sells["trade_date"].to_list(),
                    y=sells["price"].to_list(),
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="salmon"),
                )
            )
        fig.update_layout(
            template="plotly_dark",
            title=f"{pick} — price and trades",
            height=440,
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(fig, width="stretch")

        if not closed_pos.is_empty() and "close_date" in closed_pos.columns:
            daily = (
                closed_pos.group_by("close_date")
                .agg(pl.col("realized_pnl").sum().alias("daily_pnl"))
                .sort("close_date")
                .with_columns(pl.col("daily_pnl").cum_sum().alias("cumulative_pnl"))
            )
            fig_pnl = go.Figure()
            fig_pnl.add_trace(
                go.Bar(
                    x=daily["close_date"].to_list(),
                    y=daily["daily_pnl"].to_list(),
                    name="Daily realized",
                    marker_color="steelblue",
                )
            )
            fig_pnl.add_trace(
                go.Scatter(
                    x=daily["close_date"].to_list(),
                    y=daily["cumulative_pnl"].to_list(),
                    name="Cumulative",
                    mode="lines+markers",
                    yaxis="y2",
                    line=dict(color="orange", width=2),
                )
            )
            fig_pnl.update_layout(
                template="plotly_dark",
                title=f"Realized P&L ({display_ccy}, closed lots)",
                height=360,
                xaxis_title="Close date",
                yaxis=dict(title="Daily P&L"),
                yaxis2=dict(
                    title="Cumulative P&L",
                    overlaying="y",
                    side="right",
                ),
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_pnl, width="stretch")

with tab_model:
    model_weights, weight_label = load_optimization_weights(app)
    st.caption(
        f"Model weights: **{weight_label}** from `data/risk/optimization/optimal_weights.parquet` "
        "(falls back to holdings if missing)."
    )

    portfolio_value = st.number_input(
        f"Assumed portfolio value ({money_symbol(display_ccy).strip() or display_ccy})",
        min_value=1.0,
        value=float(app.tracker.default_portfolio_value),
        step=1000.0,
        key="tracker_portfolio_value",
    )

    exposure = bundle["exposure"]
    rows = []
    all_tickers = sorted(
        set(model_weights) | set(exposure["ticker"].to_list() if exposure.height else [])
    )
    for ticker in all_tickers:
        mw = model_weights.get(ticker, 0.0)
        model_pct = abs(mw) * 100
        actual_row = exposure.filter(pl.col("ticker") == ticker) if exposure.height else pl.DataFrame()
        if actual_row.height:
            notional = float(actual_row["notional"][0])
            actual_pct = abs(notional) / portfolio_value * 100 if portfolio_value else 0.0
        else:
            actual_pct = 0.0
        rows.append(
            {
                "ticker": ticker,
                "model_weight_pct": round(model_pct, 2),
                "actual_notional_pct": round(actual_pct, 2),
                "delta_pct": round(actual_pct - model_pct, 2),
            }
        )
    if rows:
        st.dataframe(pl.DataFrame(rows), width="stretch")
    else:
        st.info("No tickers to compare.")

    st.subheader("Performance comparison")
    bounds = load_comparison_bounds(app, trades)
    st.caption(f"Available data: **{bounds.min_date}** → **{bounds.max_date}**")

    preset = st.radio(
        "Period",
        ["3M", "6M", "1Y", "All"],
        horizontal=True,
        key="tracker_perf_preset",
    )
    p_start, p_end = preset_range(preset, bounds)

    c1, c2 = st.columns(2)
    with c1:
        start_d = st.date_input(
            "Start",
            value=p_start,
            min_value=bounds.min_date,
            max_value=bounds.max_date,
            key="tracker_perf_start",
        )
    with c2:
        end_d = st.date_input(
            "End",
            value=p_end,
            min_value=bounds.min_date,
            max_value=bounds.max_date,
            key="tracker_perf_end",
        )

    if start_d > end_d:
        st.error("Start date must be on or before end date.")
    else:
        dr = DateRange(start=start_d, end=end_d)
        curves = load_comparison_curves(app, trades, dr, portfolio_value)
        model_df = curves["model"]
        actual_df = curves["actual"]
        benchmark_df = curves["benchmark"]
        benchmark_label = app.tracker.benchmark_ticker or "Benchmark"
        first_trade = curves.get("first_trade_date")

        if first_trade and first_trade > start_d:
            st.caption(
                f"Actual curve begins at first trade date (**{first_trade}**); "
                "earlier dates have no ledger activity."
            )

        fig = go.Figure()
        if model_df.height:
            y_col = "equity_indexed" if "equity_indexed" in model_df.columns else "equity"
            fig.add_trace(
                go.Scatter(
                    x=model_df["date"].to_list(),
                    y=model_df[y_col].to_list(),
                    mode="lines",
                    name=f"Model ({weight_label})",
                )
            )
        if actual_df.height:
            fig.add_trace(
                go.Scatter(
                    x=actual_df["date"].to_list(),
                    y=actual_df["equity_indexed"].to_list(),
                    mode="lines",
                    name="Actual (trade ledger)",
                )
            )
        if benchmark_df.height:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["date"].to_list(),
                    y=benchmark_df["equity_indexed"].to_list(),
                    mode="lines",
                    name=f"Benchmark ({benchmark_label})",
                    line={"dash": "dot"},
                )
            )
        fig.update_layout(
            template="plotly_dark",
            title="Indexed performance (100 at range start)",
            height=420,
            xaxis_title="Date",
            yaxis_title="Index",
        )
        st.plotly_chart(fig, width="stretch")

        m1, m2, m3 = st.columns(3)
        with m1:
            if model_df.height:
                y_col = "equity_indexed" if "equity_indexed" in model_df.columns else "equity"
                ret = float(model_df[y_col][-1]) - 100.0
                st.metric(f"Model return ({preset})", f"{ret:+.1f}%")
            else:
                st.metric("Model return", "n/a")
        with m2:
            if actual_df.height:
                ret = float(actual_df["equity_indexed"][-1]) - 100.0
                st.metric(f"Actual return ({preset})", f"{ret:+.1f}%")
            else:
                st.metric("Actual return", "n/a")
        with m3:
            if benchmark_df.height:
                ret = float(benchmark_df["equity_indexed"][-1]) - 100.0
                st.metric(f"{benchmark_label} return ({preset})", f"{ret:+.1f}%")
            else:
                st.metric(f"{benchmark_label} return", "n/a")

        if model_df.is_empty() and actual_df.is_empty() and benchmark_df.is_empty():
            st.info(
                "No performance data in this window. Run the pipeline through "
                "`optimize_portfolios` and ensure market/returns exist for model; "
                "add trades and market closes for actual."
            )
