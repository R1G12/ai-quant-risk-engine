"""Portfolio analytics module."""

from __future__ import annotations

import polars as pl
import streamlit as st

from src.dashboards.core.display import format_exception
from src.dashboards.core.loaders import load_app, load_copilot_context
from src.dashboards.core.theme import apply_theme, page_header
from src.copilot.attribution.portfolio import attribution_summary
from src.risk.portfolio.holdings import load_portfolio_weights, portfolio_weighting_mode

apply_theme()
app = load_app()
page_header("Portfolio", "Holdings, attribution, and exposure")

weighting = portfolio_weighting_mode(app)
st.caption(f"Weighting mode from run profile: **{weighting}** (`configs/run.yaml` → `portfolio.weighting`)")

try:
    ctx = load_copilot_context(app)
except Exception as exc:
    st.error(f"Could not load portfolio context: {format_exception(exc)}")
    st.stop()

weights_from = str(ctx.metadata.get("weights_from", "holdings"))
st.subheader("Portfolio weights")
st.caption(
    f"Source: **{weights_from}**"
    + (
        f" (`{ctx.metadata.get('optimization_weights_path', 'optimal_weights.parquet')}`)"
        if weights_from == "optimization"
        else " (`holdings.parquet` from `aqre prepare`)"
    )
)
weights_df = pl.DataFrame(
    {
        "asset": list(ctx.weights.keys()),
        "weight": list(ctx.weights.values()),
    }
).sort("asset")
st.dataframe(weights_df, width="stretch", hide_index=True)

if weighting in ("optimised", "partial") and weights_from != "optimization":
    st.warning(
        "Optimised/partial mode is configured but optimization weights are missing. "
        "Run `dvc repro optimize_portfolios` (or `aqre run profile`) to refresh weights."
    )

st.subheader("Exposures")
st.dataframe(ctx.exposures, width="stretch")

try:
    exp = attribution_summary(ctx)
except Exception as exc:
    st.error(f"Attribution failed: {format_exception(exc)}")
else:
    st.subheader(exp.title)
    st.markdown(exp.summary)
    with st.expander("Evidence"):
        from src.dashboards.core.json_safe import json_safe

        st.json(json_safe(exp.evidence))
