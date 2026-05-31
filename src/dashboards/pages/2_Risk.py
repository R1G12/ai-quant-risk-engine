"""Risk monitoring module."""

from __future__ import annotations

import streamlit as st

from src.analytics.charts.context import DateRange
from src.analytics.charts.registry import build_chart_registry
from src.dashboards.core.charts_ui import render_chart_picker
from src.dashboards.core.loaders import load_app, load_chart_context, load_bounds
from src.dashboards.core.theme import apply_theme, page_header

apply_theme()
app = load_app()
page_header("Risk monitoring", "VaR, volatility, drawdown, correlations")

_, bounds = load_bounds(app)
dr = DateRange(start=bounds.min_date, end=bounds.max_date)
ctx = load_chart_context(app, dr)

risk_charts = [s for s in build_chart_registry() if s.group == "Risk"]
render_chart_picker(risk_charts, ctx, label="Chart", key="risk_chart_picker")
