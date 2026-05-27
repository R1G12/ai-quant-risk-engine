"""Portfolio analytics module."""

from __future__ import annotations

import streamlit as st

from src.dashboards.core.loaders import load_app, load_copilot_context
from src.dashboards.core.theme import apply_theme, page_header
from src.copilot.attribution.portfolio import attribution_summary

apply_theme()
app = load_app()
page_header("Portfolio", "Holdings, attribution, and exposure")

try:
    ctx = load_copilot_context(app)
except Exception as exc:
    st.error(f"Could not load portfolio context: {exc}")
    st.stop()

st.dataframe(ctx.exposures, width="stretch")

try:
    exp = attribution_summary(ctx)
except Exception as exc:
    st.error(f"Attribution failed: {exc}")
else:
    st.subheader(exp.title)
    st.markdown(exp.summary)
    with st.expander("Evidence"):
        from src.dashboards.core.json_safe import json_safe

        st.json(json_safe(exp.evidence))
