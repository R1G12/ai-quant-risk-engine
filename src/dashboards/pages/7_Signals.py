"""Signals: FinBERT scores, trailing stops, HMM regime, portfolio VaR."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.dashboards.core.loaders import load_app, load_kpis
from src.dashboards.core.signals_loaders import (
    load_finbert_window,
    load_latest_regime,
    load_trailing_stops_table,
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
        st.warning(
            "No FinBERT sentiment data found. Run Phase 1: "
            "`dvc repro ingest preprocess sentiment`"
        )
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
            pick = st.selectbox("Daily score history", tickers, key="signals_ticker_pick")
            sub = daily.filter(pl.col("ticker") == pick).sort("timestamp")
            if sub.height:
                line = px.line(
                    sub.to_pandas(),
                    x="timestamp",
                    y="sentiment_score",
                    markers=True,
                    title=f"{pick} — daily sentiment score (30d window)",
                )
                line.update_layout(template="plotly_dark", height=360)
                st.plotly_chart(line, width="stretch")

with tab_stops:
    if summary is None:
        st.warning("Trailing stops require FinBERT summary — run the sentiment pipeline first.")
    else:
        stops_df = load_trailing_stops_table(summary)
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
        tail = regimes_df.tail(120)
        reg_plot = px.scatter(
            tail.to_pandas(),
            x="timestamp",
            y="regime_label",
            title="HMM regime history (last 120 observations)",
        )
        reg_plot.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(reg_plot, width="stretch")

    if kpis.var_95 is None:
        st.info("Run Phase 3 VaR: `dvc repro generate_var_metrics`")
