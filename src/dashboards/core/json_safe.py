"""Make copilot evidence safe for Streamlit st.json."""

from __future__ import annotations

from typing import Any

from src.utils.paths import display_path, sanitize_display_text


def _sanitize_str(s: str) -> str:
    cleaned = sanitize_display_text(s)
    if cleaned != s:
        return cleaned
    try:
        rel = display_path(s)
        if rel != s and (":" in s or s.startswith("/") or s.startswith("\\")):
            return rel
    except Exception:
        pass
    return s


def json_safe(value: object) -> object:
    """Recursively convert numpy scalars and other types for JSON display."""
    if isinstance(value, str):
        return _sanitize_str(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return _sanitize_str(str(value))
