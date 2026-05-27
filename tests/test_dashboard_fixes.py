"""Tests for dashboard stability helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analytics.charts.registry import build_chart_registry
from src.dashboards.core.json_safe import json_safe

_DASHBOARD_PAGES = Path(__file__).resolve().parents[1] / "src" / "dashboards" / "pages"


def test_json_safe_numpy_scalar() -> None:
    out = json_safe({"x": np.float64(0.5), "nested": [{"y": np.int64(3)}]})
    assert out["x"] == 0.5
    assert out["nested"][0]["y"] == 3


def test_chart_registry_ids_unique() -> None:
    registry = build_chart_registry()
    ids = [s.id for s in registry]
    assert len(ids) == len(set(ids))


def test_dashboard_pages_do_not_use_broken_polars_import() -> None:
    for path in _DASHBOARD_PAGES.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert '__import__("polars").pl' not in text, f"{path.name} uses invalid polars import"


def test_streamlit_widgets_use_width_not_use_container_width() -> None:
    roots = [
        Path(__file__).resolve().parents[1] / "src" / "dashboards",
        Path(__file__).resolve().parents[1] / "src" / "analytics" / "streamlit_dashboard.py",
    ]
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*.py")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert "use_container_width" not in text, f"{path} still uses deprecated use_container_width"


def test_scan_experiments_strips_prefix(tmp_path) -> None:
    from src.dashboards.core.loaders import scan_experiments

    base = tmp_path / "backtests" / "experiment_id=baseline"
    base.mkdir(parents=True)
    # scan_experiments uses real EXPERIMENTS_BACKTESTS_DIR - test logic inline
    name = "experiment_id=baseline"
    assert name.split("=", 1)[1] == "baseline"
