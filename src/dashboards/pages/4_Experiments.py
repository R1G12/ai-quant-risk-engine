"""Experiment tracking dashboard."""

from __future__ import annotations

import polars as pl
import streamlit as st

from src.dashboards.core.loaders import load_app, scan_experiments
from src.dashboards.core.theme import apply_theme, page_header
from src.utils.paths import RESEARCH_COMPARISONS_PATH

apply_theme()
app = load_app()
page_header("Experiments", "DVC experiment manifests and comparisons")

st.caption(f"Active experiment: `{app.research.meta.experiment_id}`")
exp_df = scan_experiments()
if exp_df.height:
    st.dataframe(exp_df, width="stretch")
else:
    st.info("No experiment manifests under `experiments/` yet. Run Phase 4 with DVCLive enabled.")

if RESEARCH_COMPARISONS_PATH.is_file():
    st.subheader("Comparison table")
    st.dataframe(pl.read_parquet(RESEARCH_COMPARISONS_PATH), width="stretch")

st.markdown(
    "**Sweep example:** `dvc exp run generate_simulations -S research.simulation.n_paths=5000`"
)
