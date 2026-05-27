"""Pipeline health and data quality monitoring."""

from __future__ import annotations

import streamlit as st

from src.dashboards.core.loaders import load_app
from src.dashboards.core.theme import apply_theme, page_header
from src.monitoring.health.checks import run_health_checks
from src.monitoring.quality.validators import run_quality_checks

apply_theme()
app = load_app()
page_header("Monitoring", "Artifact freshness, lineage, and data quality")

health = run_health_checks(app)
quality = run_quality_checks(app)

st.caption(
    "Health = artifact files present on disk. Quality = schema/coverage inside `risk_dataset`. "
    "Missing Phase 3/4 outputs show as warnings until you run `dvc repro`."
)

st.subheader("Health checks")
for check in health:
    icon = "[OK]" if check.ok else "[missing]"
    st.markdown(f"{icon} **{check.name}**: {check.message}")

st.subheader("Quality checks")
for check in quality:
    icon = "[OK]" if check.ok else "[check]"
    st.markdown(f"{icon} **{check.name}**: {check.message}")
