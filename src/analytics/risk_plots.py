"""Phase 3 Plotly risk visualizations (delegates to chart registry)."""

from __future__ import annotations

from pathlib import Path

from src.analytics.charts.registry import build_chart_registry, write_standalone_charts
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.paths import ANALYTICS_RESEARCH_DIR, ANALYTICS_RISK_DIR, ensure_dir

LOGGER = get_logger(__name__)


def write_risk_dashboard(output_dir: Path | None = None) -> None:
    """Write risk-group standalone HTML plots."""
    app = load_app_config()
    out = output_dir or ANALYTICS_RISK_DIR
    ensure_dir(out)
    registry = [s for s in build_chart_registry() if s.group == "Risk"]
    write_standalone_charts(registry, app, ANALYTICS_RESEARCH_DIR, out)
    LOGGER.info("Risk dashboard written", extra={"dir": str(out)})
