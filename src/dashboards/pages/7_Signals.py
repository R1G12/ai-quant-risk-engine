"""Signals: FinBERT scores, trailing stops, HMM regime, portfolio VaR."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.dashboards.core.loaders import load_app, load_kpis
from src.dashboards.core.signals_loaders import (
    HMM_REGIME_LABELS,
    RECENT_SCORE_HISTORY_DAYS,
    REGIME_HISTORY_OBS,
    finbert_empty_reason,
    finbert_recent_days_panel,
    load_finbert_window,
    load_latest_regime,
    load_trailing_stops_table,
    regime_day_counts,
    regime_history_window,
    regimes_missing_in_window,
)
from src.dashboards.core.theme import apply_theme, page_header

apply_theme()
app = load_app()
page_header(
    "Signals",
    "FinBERT (30d), sentiment-derived trailing stops, HMM regime, portfolio VaR 95%",
)

st.caption(
    "Sentiment score = mean(positive) − mean(negative) on FinBERT labels over the last 30 days. "
    "Stop levels match the trading notebook policy (bull / neutral / bear)."
)

summary, daily = load_finbert_window(app, window_days=30)
kpis = load_kpis(app, None)
regime_label, regimes_df = load_latest_regime()

tab_finbert, tab_stops, tab_risk = st.tabs(["FinBERT", "Trailing stops", "Regime & VaR"])

with tab_finbert:
    if summary is None:
        reason, missing_tickers = finbert_empty_reason(app, window_days=30)
        st.warning(reason or "No FinBERT sentiment data for the current holdings.")
        if missing_tickers:
            st.caption(f"Tickers with no mapped articles: {', '.join(missing_tickers)}")
    else:
        fig = px.bar(
            summary.to_pandas(),
            x="ticker",
            y="sentiment_score",
            color="sentiment_score",
            color_continuous_scale="RdYlGn",
            title="30-day FinBERT sentiment score by ticker",
            labels={"sentiment_score": "Score (-1 bearish → +1 bullish)"},
        )
        fig.update_layout(template="plotly_dark", height=420, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            summary.select(
                "ticker",
                "sentiment_score",
                "bullish_ratio",
                "negative_ratio",
                "article_count",
                "avg_confidence",
            ),
            width="stretch",
        )

        if daily is not None and daily.height:
            tickers = summary["ticker"].to_list()
            days = st.slider(
                "Recent sentiment window (days)",
                min_value=1,
                max_value=7,
                value=RECENT_SCORE_HISTORY_DAYS,
                step=1,
                key="signals_recent_days",
            )
            panel, stale_meta = finbert_recent_days_panel(daily, tickers, n_days=days)
            if panel.height:
                st.subheader(f"Last {days} days — all tickers")
                st.caption(
                    "Shared calendar window for every holding. "
                    "Scores without fresh articles are carried forward from the last day with news."
                )

                if not stale_meta.is_empty():
                    no_data = stale_meta.filter(pl.col("no_data") == True)  # noqa: E712
                    if no_data.height:
                        for row in no_data.iter_rows(named=True):
                            st.error(
                                f"**{row['ticker']}** — no FinBERT articles in the {days}-day window; "
                                "not shown on the chart."
                            )

                    stale_only = stale_meta.filter(pl.col("has_stale") == True)  # noqa: E712
                    if stale_only.height:
                        stale_table = (
                            stale_only.select(
                                "ticker",
                                pl.col("last_article_day")
                                .cast(pl.Utf8)
                                .fill_null("—")
                                .alias("Last news day"),
                            )
                        )
                        st.dataframe(stale_table, width="stretch")
                        st.caption(
                            "Forward-filled scores: each listed ticker repeats the latest available "
                            "sentiment score on days without new articles."
                        )

                panel_pd = panel.to_pandas()
                line = px.line(
                    panel_pd,
                    x="timestamp",
                    y="sentiment_score",
                    color="ticker",
                    markers=False,
                    title=(
                        f"FinBERT sentiment — last {days} calendar days "
                        "(all holdings)"
                    ),
                    labels={"sentiment_score": "Score (-1 bearish → +1 bullish)"},
                )
                line.update_traces(line=dict(width=2.5))
                ticker_colors = {
                    str(trace.name): trace.line.color
                    for trace in line.data
                    if trace.name and trace.line.color is not None
                }
                for ticker in tickers:
                    color = ticker_colors.get(ticker, "#888888")
                    t_fresh = panel_pd[
                        (panel_pd["ticker"] == ticker) & ~panel_pd["is_forward_filled"]
                    ]
                    if not t_fresh.empty:
                        line.add_trace(
                            go.Scatter(
                                x=t_fresh["timestamp"],
                                y=t_fresh["sentiment_score"],
                                mode="markers",
                                marker=dict(size=10, color=color),
                                legendgroup=ticker,
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                for ticker in tickers:
                    color = ticker_colors.get(ticker, "#888888")
                    t_fill = panel_pd[
                        (panel_pd["ticker"] == ticker) & panel_pd["is_forward_filled"]
                    ]
                    if not t_fill.empty:
                        line.add_trace(
                            go.Scatter(
                                x=t_fill["timestamp"],
                                y=t_fill["sentiment_score"],
                                mode="markers",
                                marker=dict(
                                    symbol="circle-open",
                                    size=16,
                                    color=color,
                                    line=dict(width=3, color=color),
                                ),
                                legendgroup=ticker,
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                y_vals = panel_pd["sentiment_score"]
                y_min, y_max = float(y_vals.min()), float(y_vals.max())
                y_pad = max(0.08, (y_max - y_min) * 0.2) if y_max != y_min else 0.12
                line.update_layout(
                    template="plotly_dark",
                    height=max(560, 80 * len(tickers)),
                    margin=dict(l=48, r=32, t=56, b=48),
                    yaxis=dict(
                        range=[y_min - y_pad, y_max + y_pad],
                        tickformat=".2f",
                        title="Score (-1 bearish → +1 bullish)",
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                )
                st.plotly_chart(line, width="stretch")

                show_cols = [
                    "ticker",
                    "day",
                    "sentiment_score",
                    "is_forward_filled",
                    "score_as_of",
                ]
                table = panel.select(show_cols).with_columns(
                    pl.col("day").cast(pl.Utf8).alias("day"),
                    pl.col("score_as_of").cast(pl.Utf8).alias("score_as_of"),
                )
                st.dataframe(table, width="stretch")

with tab_stops:
    if summary is None:
        st.warning("Trailing stops require FinBERT summary — run the sentiment pipeline first.")
    else:
        stops_df = load_trailing_stops_table(summary, app)
        st.dataframe(stops_df, width="stretch")

        heat = stops_df.select(
            "ticker",
            "stop_1_level",
            "stop_2_level",
            "stop_3_level",
        ).to_pandas()
        heat_long = heat.melt(id_vars=["ticker"], var_name="tranche", value_name="level")
        hm = px.imshow(
            heat_long.pivot(index="ticker", columns="tranche", values="level"),
            labels=dict(color="Drawdown trigger"),
            title="Stop levels by ticker (drawdown from peak, negative %)",
            color_continuous_scale="Reds_r",
            aspect="auto",
        )
        hm.update_layout(template="plotly_dark", height=max(320, 40 * len(heat)))
        st.plotly_chart(hm, width="stretch")

        st.subheader("Tranche 1 (tightest) level")
        t1 = px.bar(
            stops_df.to_pandas(),
            x="ticker",
            y="tranche_1_level",
            color="regime_tag",
            title="Tranche 1 stop level per ticker",
            labels={"tranche_1_level": "Level"},
        )
        t1.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(t1, width="stretch")

        pick2 = st.selectbox("Stop ladder", stops_df["ticker"].to_list(), key="signals_ladder_pick")
        row = stops_df.filter(pl.col("ticker") == pick2).row(0, named=True)
        ladder = go.Figure()
        for i, key in enumerate(("stop_1_level", "stop_2_level", "stop_3_level"), start=1):
            lvl = float(row[key])
            ladder.add_trace(
                go.Scatter(
                    x=[i],
                    y=[lvl],
                    mode="markers+text",
                    text=[f"{lvl:.0%}"],
                    textposition="top center",
                    name=f"Stop {i}",
                )
            )
        ladder.update_layout(
            title=f"{pick2} — 3-tranche stop ladder ({row['regime_tag']})",
            template="plotly_dark",
            yaxis_title="Drawdown trigger",
            xaxis_title="Tranche",
            height=360,
        )
        st.plotly_chart(ladder, width="stretch")

with tab_risk:
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Current HMM regime",
            regime_label if regime_label else "n/a",
        )
    with c2:
        st.metric(
            "Portfolio VaR 95%",
            f"{kpis.var_95:.2%}" if kpis.var_95 is not None else "n/a",
        )

    if regime_label is None:
        st.info("Run Phase 3 portfolio metrics: `dvc repro generate_portfolio_metrics`")
    elif regimes_df is not None and regimes_df.height > 1:
        win = regime_history_window(regimes_df, n_obs=REGIME_HISTORY_OBS)
        reg_plot = px.scatter(
            win.to_pandas(),
            x="timestamp",
            y="regime_label",
            title=f"HMM regime history (last {REGIME_HISTORY_OBS} observations)",
            category_orders={"regime_label": list(HMM_REGIME_LABELS)},
        )
        reg_plot.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(reg_plot, width="stretch")
        missing = regimes_missing_in_window(win)
        if missing:
            st.caption(
                f"No days assigned to regime(s) in this window: {', '.join(missing)} "
                "(y-axis still shows all three HMM labels)."
            )
        st.subheader("Days per regime")
        st.dataframe(regime_day_counts(win), width="stretch")

    if kpis.var_95 is None:
        st.info("Run Phase 3 VaR: `dvc repro generate_var_metrics`")
