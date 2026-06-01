"""User-facing text helpers for Streamlit (no absolute machine paths)."""

from __future__ import annotations

from src.utils.paths import sanitize_display_text


def format_exception(exc: BaseException) -> str:
    """Error message safe to show in the UI (project-relative paths only)."""
    return sanitize_display_text(str(exc))
