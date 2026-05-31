"""Make copilot evidence safe for Streamlit st.json."""

from __future__ import annotations

from typing import Any


def json_safe(value: object) -> object:
    """Recursively convert numpy scalars and other types for JSON display."""
    if value is None or isinstance(value, (str, int, float, bool)):
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
    return str(value)
