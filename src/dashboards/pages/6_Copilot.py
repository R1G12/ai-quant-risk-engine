"""AI portfolio copilot interface."""

from __future__ import annotations

import streamlit as st

from src.dashboards.core.loaders import load_app, load_copilot_context
from src.dashboards.core.theme import apply_theme, page_header
from src.copilot.reasoning.engine import CopilotEngine
from src.copilot.summarization.report import generate_portfolio_summary
from src.dashboards.core.json_safe import json_safe

apply_theme()
app = load_app()
page_header("Portfolio copilot", "Auditable Q&A · human-in-the-loop")

ctx = load_copilot_context(app)
st.markdown(generate_portfolio_summary(ctx))

SUGGESTIONS = [
    "Why did portfolio risk increase?",
    "Which assets contribute most to VaR?",
    "What drove today's drawdown?",
    "What is the current volatility regime?",
    "Which positions are sentiment-sensitive?",
    "What changed in the correlation structure?",
]

if "copilot_q" not in st.session_state:
    st.session_state.copilot_q = ""

st.subheader("Suggested questions")
cols = st.columns(2)
for i, q in enumerate(SUGGESTIONS):
    if cols[i % 2].button(q, key=f"suggest_{i}"):
        st.session_state.copilot_q = q
        st.rerun()

question = st.text_area("Your question", key="copilot_q", height=80)
if st.button("Ask copilot", type="primary") and question.strip():
    resp = CopilotEngine(app).ask(question.strip(), ctx)
    st.markdown(f"### {resp.title}")
    st.markdown(resp.answer)
    st.caption(f"Confidence: {resp.confidence} · Human review: {resp.requires_human_review}")
    with st.expander("Evidence JSON (audit)"):
        st.json(json_safe(resp.evidence))

st.warning(
    "Copilot outputs are **decision support only**. No automated execution. "
    "Validate all recommendations with risk committee process."
)
