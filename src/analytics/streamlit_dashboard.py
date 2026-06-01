"""Interactive Streamlit dashboard for research and risk charts."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import streamlit as st

from src.analytics.charts.context import ChartContext, DateRange, filter_by_date_range
from src.analytics.charts.registry import build_chart_registry
from src.analytics.dashboard_kpis import compute_window_kpis, equity_csv_bytes
from src.analytics.data_bounds import DataBounds, default_last_year_range, detect_data_bounds, slider_bounds
from src.analytics.live_market_cache import (
    fetch_tickers_market,
    invalidate_live_cache,
    is_live_ready,
    live_fetch_error,
    load_live_prices,
    start_background_fetch,
)
from src.dashboards.core.display import format_exception
from src.analytics.ticker_utils import (
    LIVE_TICKER_CHART_IDS,
    chart_uses_pipeline_only,
    parse_ticker_input,
    pipeline_ticker_warning_markdown,
    pipeline_tickers,
)
from src.utils.config import load_app_config


def _init_session() -> None:
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = "sample"
    if "live_started" not in st.session_state:
        st.session_state.live_started = False
    if "date_preset" not in st.session_state:
        st.session_state.date_preset = "1Y"
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = ""
    if "custom_ticker_df" not in st.session_state:
        st.session_state.custom_ticker_df = None
    if "custom_ticker_error" not in st.session_state:
        st.session_state.custom_ticker_error = None


def _preset_range(preset: str, bounds: DataBounds) -> tuple[date, date]:
    """Map preset label to [start, end] clipped to available data."""
    end = min(bounds.max_date, date.today())
    if preset == "All":
        return bounds.min_date, bounds.max_date
    if preset == "YTD":
        start = date(end.year, 1, 1)
        return max(bounds.min_date, start), end
    days = {"1Y": 365, "6M": 183, "3M": 92}.get(preset, 365)
    start = max(bounds.min_date, end - timedelta(days=days))
    return start, end


def _stale_data_banner(bounds: DataBounds) -> None:
    if bounds.max_date < date.today() - timedelta(days=30):
        st.warning(
            f"**Sample** artifacts on disk span **{bounds.min_date}** to **{bounds.max_date}**. "
            "Charts and the date slider use this range until you refresh pipeline data. "
            "See **Refresh sample data** in the sidebar, or use **Live** once yfinance succeeds."
        )


def _scenario_table(app, chart_id: str) -> None:
    from src.utils.paths import RESEARCH_SCENARIOS_DIR, RESEARCH_STRESS_DIR

    exp = app.research.meta.experiment_id
    if chart_id == "stress_dashboard":
        path = RESEARCH_STRESS_DIR / f"experiment_id={exp}" / "stress_metrics.parquet"
    else:
        path = RESEARCH_SCENARIOS_DIR / f"experiment_id={exp}" / "scenario_comparison.parquet"
    if path.is_file():
        st.dataframe(pl.read_parquet(path), width="stretch")


def _resolve_live_price_df(custom_tickers: list[str]) -> pl.DataFrame | None:
    """Custom fetch takes priority; else background cache."""
    if st.session_state.custom_ticker_df is not None and custom_tickers:
        df = st.session_state.custom_ticker_df
        if "ticker" in df.columns:
            return df.filter(pl.col("ticker").is_in(custom_tickers))
        return df
    df = load_live_prices()
    if df is None:
        return None
    if custom_tickers and "ticker" in df.columns:
        sub = df.filter(pl.col("ticker").is_in(custom_tickers))
        return sub if sub.height else df
    return df


def _live_price_chart(dr: DateRange | None, tickers: list[str], *, title: str) -> None:
    import plotly.graph_objects as go

    df = _resolve_live_price_df(tickers)
    if df is None or df.height == 0:
        err = st.session_state.custom_ticker_error
        if err:
            st.error(f"Could not load prices: {err}")
        else:
            st.info("No live prices yet. Enter tickers and click **Fetch tickers**, or wait for background load.")
        return
    df = filter_by_date_range(df, "timestamp", dr)
    if df.height == 0:
        st.info("No live rows in selected date range.")
        return
    if "close" not in df.columns:
        st.info("Live cache missing close prices.")
        return
    fig = go.Figure()
    for ticker in tickers:
        sub = df.filter(pl.col("ticker") == ticker).sort("timestamp")
        if sub.height == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["close"],
                mode="lines",
                name=ticker,
            )
        )
    if not fig.data:
        st.warning(f"No rows for: {', '.join(tickers)}. Try **Fetch tickers** again.")
        return
    fig.update_layout(title=title, template="plotly_dark", height=480)
    st.plotly_chart(fig, width="stretch")


def _show_pipeline_ticker_warning(app, spec, custom_tickers: list[str]) -> None:
    st.warning(pipeline_ticker_warning_markdown(app, custom_tickers))


def main() -> None:
    st.set_page_config(page_title="AI Quant Risk Dashboard", layout="wide", initial_sidebar_state="expanded")
    _init_session()
    app = load_app_config()
    raw_bounds = detect_data_bounds(app.research.meta.experiment_id)
    live_ready = is_live_ready()
    bounds = slider_bounds(raw_bounds, include_today_if_live=live_ready)
    registry = build_chart_registry()
    groups = sorted({s.group for s in registry})
    default_ticker_str = ", ".join(pipeline_tickers(app))

    if not st.session_state.live_started:
        start_background_fetch(app)
        st.session_state.live_started = True

    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a; color: #e2e8f0; }
        [data-testid="stSidebar"] { background-color: #1e293b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.title("AI Quant Risk — Research Dashboard")
        st.caption(f"Experiment: `{app.research.meta.experiment_id}`")
    with col_h2:
        mode = st.session_state.data_mode
        badge = "Live" if mode == "live" else "Sample"
        st.metric("Data mode", badge)
    with col_h3:
        st.metric("Data span", f"{bounds.min_date} → {bounds.max_date}")

    _stale_data_banner(raw_bounds)

    if not live_ready and st.session_state.custom_ticker_df is None:
        err = live_fetch_error()
        msg = "Loading live market data in background…" if err is None else f"Live fetch: {err}"
        st.progress(0.35, text=msg)
    elif live_ready or st.session_state.custom_ticker_df is not None:
        st.success("Live prices available — use tickers in the sidebar and **Use live data** for price charts.")

    with st.sidebar:
        st.header("Controls")

        st.subheader("Tickers (live prices)")
        st.caption("Enter any Yahoo symbols (comma-separated). **Fetch** loads ~1Y of OHLCV via yfinance.")
        if not st.session_state.ticker_input:
            st.session_state.ticker_input = default_ticker_str
        ticker_text = st.text_area(
            "Tickers",
            value=st.session_state.ticker_input,
            height=68,
            placeholder="AAPL, MSFT, TSLA, NVDA",
            key="ticker_text_area",
        )
        custom_tickers = parse_ticker_input(ticker_text)
        st.session_state.ticker_input = ticker_text

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Fetch tickers", width="stretch"):
                if not custom_tickers:
                    st.session_state.custom_ticker_error = "Enter at least one ticker."
                else:
                    with st.spinner(f"Fetching {len(custom_tickers)} tickers…"):
                        try:
                            st.session_state.custom_ticker_df = fetch_tickers_market(
                                custom_tickers, app.market
                            )
                            st.session_state.custom_ticker_error = None
                        except Exception as exc:
                            st.session_state.custom_ticker_df = None
                            st.session_state.custom_ticker_error = str(exc)
                st.rerun()
        with col_b:
            if st.button("Reset tickers", width="stretch"):
                st.session_state.ticker_input = default_ticker_str
                st.session_state.custom_ticker_df = None
                st.session_state.custom_ticker_error = None
                st.rerun()

        if st.session_state.custom_ticker_error:
            st.caption(f"Last error: {st.session_state.custom_ticker_error}")

        display_tickers = custom_tickers or pipeline_tickers(app)
        st.caption(f"Showing: **{', '.join(display_tickers)}**")

        st.divider()
        group = st.selectbox("Chart group", groups)
        charts_in_group = [s for s in registry if s.group == group]
        by_id = {s.id: s for s in charts_in_group}
        chosen_id = st.selectbox(
            "Chart",
            list(by_id.keys()),
            format_func=lambda cid: by_id[cid].title,
            key=f"legacy_chart_{group}",
        )
        spec = by_id[chosen_id]

        st.caption(
            "**Date preset** sets the slider to a quick window (last 3M / 6M / 1Y, year-to-date, or all data on disk). "
            "You can still fine-tune with the slider below."
        )
        preset = st.radio("Date preset", ["1Y", "6M", "3M", "YTD", "All"], horizontal=True, key="preset_radio")
        if st.session_state.date_preset != preset:
            st.session_state.date_preset = preset
            start, end = _preset_range(preset, bounds)
            st.session_state.dr_start = start
            st.session_state.dr_end = end

        default_start, default_end = _preset_range(st.session_state.date_preset, bounds)
        if preset == "1Y" and bounds.max_date == bounds.min_date:
            default_start, default_end = default_last_year_range(bounds)

        dr_start = st.session_state.get("dr_start", default_start)
        dr_end = st.session_state.get("dr_end", default_end)
        dr_tuple = st.slider(
            "Date range",
            min_value=bounds.min_date,
            max_value=bounds.max_date,
            value=(dr_start, dr_end),
            format="YYYY-MM-DD",
        )
        st.session_state.dr_start, st.session_state.dr_end = dr_tuple
        date_range = DateRange(start=dr_tuple[0], end=dr_tuple[1])

        st.divider()
        st.subheader("Data source")
        live_ok = live_ready or st.session_state.custom_ticker_df is not None
        use_live = st.button(
            "Use live data",
            disabled=not live_ok,
            help="Shows yfinance closes for your tickers on Backtest equity / Rolling Sharpe charts.",
        )
        if st.button("Use sample data"):
            st.session_state.data_mode = "sample"
        if use_live:
            st.session_state.data_mode = "live"
        if st.button("Retry live fetch"):
            invalidate_live_cache()
            st.session_state.live_started = False
            st.rerun()

        with st.expander("Refresh sample data (~1 year)"):
            st.code(
                '$env:MARKET_SOURCE = "sample"\n'
                "dvc repro ingest_market_data clean_market_data generate_returns merge_features\n"
                "# then risk + research stages as needed",
                language="powershell",
            )
            st.caption(
                f"Config targets {app.market.start_date} → {app.market.end_date} for **new** pipeline runs. "
                f"On-disk sample currently ends {raw_bounds.max_date}."
            )

    ctx = ChartContext(app=app, date_range=date_range if spec.date_filterable else None)
    pipe_only = chart_uses_pipeline_only(spec, data_mode=st.session_state.data_mode)
    tickers_changed = set(custom_tickers or pipeline_tickers(app)) != set(pipeline_tickers(app))

    try:
        kpis = compute_window_kpis(app, date_range)
    except Exception as exc:
        st.error(f"KPI calculation failed: {format_exception(exc)}")
        kpis = None

    main_col, side_col = st.columns([3, 1])
    with side_col:
        st.subheader("KPIs (window)")
        if kpis:
            if kpis.sharpe is not None:
                st.metric("Sharpe", f"{kpis.sharpe:.2f}")
            if kpis.max_drawdown is not None:
                st.metric("Max drawdown", f"{kpis.max_drawdown:.1%}")
            if kpis.var_95 is not None:
                st.metric("VaR 95%", f"{kpis.var_95:.2%}")
        st.markdown(f"**{spec.title}**")
        st.caption(spec.description)
        with st.expander("How to read this chart"):
            st.write(spec.description)
            if not spec.date_filterable:
                st.info("This chart is not filtered by the date slider (full simulation/stress horizon).")
        if st.session_state.data_mode == "live" and spec.id in LIVE_TICKER_CHART_IDS:
            st.success(f"Live closes for: {', '.join(display_tickers)}")
        elif tickers_changed or custom_tickers:
            st.caption("KPIs are from the **pipeline backtest**, not your custom tickers.")

    with main_col:
        show_warning = pipe_only and (tickers_changed or bool(custom_tickers))
        if show_warning:
            _show_pipeline_ticker_warning(app, spec, custom_tickers or display_tickers)

        if (
            st.session_state.data_mode == "live"
            and spec.id in LIVE_TICKER_CHART_IDS
            and display_tickers
        ):
            _live_price_chart(
                date_range if spec.date_filterable else None,
                display_tickers,
                title=f"Live closes — {', '.join(display_tickers)}",
            )
            if spec.id == "equity_curve":
                st.caption(
                    "Portfolio equity below is still from the **last DVC backtest** (pipeline tickers). "
                    "Use sample mode to view only the backtest chart."
                )
                try:
                    fig = spec.builder(ctx)
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                except Exception as exc:
                    st.error(f"Backtest chart failed: {format_exception(exc)}")
        else:
            try:
                fig = spec.builder(ctx)
            except Exception as exc:
                st.error(f"Chart build failed: {format_exception(exc)}")
                fig = None
            if fig is None:
                st.warning("No data for this chart. Run `dvc repro` through research stages.")
            else:
                st.plotly_chart(fig, width="stretch")

        if spec.id in ("stress_dashboard", "scenario_comparison"):
            _scenario_table(app, spec.id)

        csv_bytes = equity_csv_bytes(app, date_range) if spec.id == "equity_curve" else None
        if csv_bytes and st.session_state.data_mode != "live":
            st.download_button(
                "Download filtered equity CSV",
                data=csv_bytes,
                file_name="equity_curve_filtered.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
