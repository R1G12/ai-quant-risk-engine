"""User-facing error handling for dashboard pages."""

from __future__ import annotations

import streamlit as st


def show_data_unavailable(feature: str, hint: str) -> None:
    st.warning(f"**{feature}** is not available yet. {hint}")
