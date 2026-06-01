"""Streamlit chart picker helpers (stable across reruns)."""

from __future__ import annotations

import streamlit as st

from src.dashboards.core.display import format_exception
from src.analytics.charts.base import ChartSpec
from src.analytics.charts.context import ChartContext


def render_chart_picker(
    charts: list[ChartSpec],
    ctx: ChartContext,
    *,
    label: str = "Chart",
    key: str = "chart_picker",
) -> None:
    """Render selectbox + plotly chart. Uses chart ids (not ChartSpec objects) as widget values."""
    if not charts:
        st.warning("No charts in this group. Run the relevant DVC pipeline stages first.")
        return

    by_id = {s.id: s for s in charts}
    chosen_id = st.selectbox(
        label,
        list(by_id.keys()),
        format_func=lambda cid: by_id[cid].title,
        key=key,
    )
    spec = by_id[chosen_id]
    try:
        fig = spec.builder(ctx)
    except Exception as exc:
        st.error(f"Chart `{spec.id}` failed to build: {format_exception(exc)}")
        return

    if fig is None:
        st.warning("No data for this chart. Run `dvc repro` for the required pipeline stage.")
    else:
        st.plotly_chart(fig, width="stretch")
    st.caption(spec.description)
