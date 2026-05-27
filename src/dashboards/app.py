"""Institutional platform dashboard — multi-page Streamlit entry."""

from __future__ import annotations

import streamlit as st

from datetime import date, timedelta

from src.dashboards.core.json_safe import json_safe
from src.dashboards.core.theme import apply_theme, page_header
from src.dashboards.core.loaders import load_app, load_kpis, load_bounds
from src.copilot.reasoning.engine import CopilotEngine
from src.copilot.summarization.report import generate_portfolio_summary
from src.copilot.context.builder import build_portfolio_context

st.set_page_config(
    page_title="AQRE Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
app = load_app()

page_header(
    "AI Quant Risk — Portfolio Intelligence Platform",
    f"Experiment `{app.research.meta.experiment_id}` · Phase 5 decision support",
)

raw_bounds, bounds = load_bounds(app)

if raw_bounds.max_date < date.today() - timedelta(days=30):
    st.warning(
        f"Sample artifacts span **{raw_bounds.min_date}** to **{raw_bounds.max_date}**. "
        "Refresh pipeline data or use the legacy dashboard's live yfinance mode for recent prices."
    )

try:
    kpis = load_kpis(app, None)
except Exception as exc:
    st.error(f"KPI calculation failed: {exc}")
    from src.analytics.dashboard_kpis import WindowKpis

    kpis = WindowKpis(None, None, None)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Sharpe (window)", f"{kpis.sharpe:.2f}" if kpis.sharpe is not None else "n/a")
with c2:
    st.metric("Max drawdown", f"{kpis.max_drawdown:.1%}" if kpis.max_drawdown is not None else "n/a")
with c3:
    st.metric("VaR 95%", f"{kpis.var_95:.2%}" if kpis.var_95 is not None else "n/a")
with c4:
    st.metric("Data span", f"{bounds.min_date} -> {bounds.max_date}")

st.info(
    "Use the **sidebar pages** for Portfolio, Risk, Simulations, Experiments, Monitoring, and Copilot. "
    "Legacy chart explorer: `streamlit run src/analytics/streamlit_dashboard.py`"
)

st.subheader("Executive summary")
try:
    ctx = build_portfolio_context(app)
    st.markdown(generate_portfolio_summary(ctx))
except Exception as exc:
    st.error(f"Could not build portfolio summary: {exc}")
    ctx = None

st.subheader("Quick copilot")
question = st.text_input(
    "Ask the portfolio copilot",
    placeholder="Why did portfolio risk increase?",
)
if question and ctx is not None:
    try:
        resp = CopilotEngine(app).ask(question, ctx)
        st.markdown(f"**{resp.title}** (confidence: {resp.confidence})")
        st.markdown(resp.answer)
        if resp.requires_human_review:
            st.caption("Human validation required before acting on this output.")
        with st.expander("Evidence (audit trail)"):
            st.json(json_safe(resp.evidence))
    except Exception as exc:
        st.error(f"Copilot error: {exc}")
elif question and ctx is None:
    st.warning("Load portfolio context first (run Phase 2-3 pipeline).")
