"""Institutional dashboard styling."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b1220; color: #e2e8f0; }
        [data-testid="stSidebar"] { background-color: #111827; }
        .metric-card { padding: 0.5rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
