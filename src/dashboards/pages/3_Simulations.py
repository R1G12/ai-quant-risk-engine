"""Monte Carlo and stress test visualizations."""

from __future__ import annotations

import streamlit as st

import polars as pl

from src.analytics.charts.context import ChartContext
from src.analytics.charts.registry import build_chart_registry
from src.dashboards.core.charts_ui import render_chart_picker
from src.dashboards.core.loaders import load_app
from src.dashboards.core.theme import apply_theme, page_header
from src.utils.paths import RESEARCH_SCENARIOS_DIR, RESEARCH_STRESS_DIR

apply_theme()
app = load_app()
page_header("Simulations & stress", "MC paths, stress VaR, scenarios")

ctx = ChartContext(app=app, date_range=None)
groups = ("Simulation", "Stress")
charts = [s for s in build_chart_registry() if s.group in groups]
render_chart_picker(charts, ctx, label="Chart", key="sim_chart_picker")

exp = app.research.meta.experiment_id
stress_path = RESEARCH_STRESS_DIR / f"experiment_id={exp}" / "stress_metrics.parquet"
scen_path = RESEARCH_SCENARIOS_DIR / f"experiment_id={exp}" / "scenario_comparison.parquet"
if stress_path.is_file():
    st.subheader("Stress metrics")
    st.dataframe(pl.read_parquet(stress_path), width="stretch")
if scen_path.is_file():
    st.subheader("Scenario comparison")
    st.dataframe(pl.read_parquet(scen_path), width="stretch")
