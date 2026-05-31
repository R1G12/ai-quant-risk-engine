"""Phase 4 Plotly research visualizations (delegates to chart registry)."""

from __future__ import annotations

from pathlib import Path

from src.analytics.charts.registry import build_chart_registry, write_standalone_charts
from src.utils.config import AppConfig, load_app_config
from src.utils.logger import get_logger
from src.utils.paths import ANALYTICS_RESEARCH_DIR, ANALYTICS_RISK_DIR, ensure_dir

LOGGER = get_logger(__name__)


def write_research_dashboard(
    output_dir: Path | None = None,
    *,
    experiment_id: str = "baseline",
    app: AppConfig | None = None,
) -> dict[str, object]:
    """Write research-group standalone HTML charts."""
    app = app or load_app_config()
    out = output_dir or ANALYTICS_RESEARCH_DIR
    ensure_dir(out)
    registry = [s for s in build_chart_registry() if s.group in ("Backtest", "Simulation", "Stress")]
    figures = write_standalone_charts(registry, app, out, ANALYTICS_RISK_DIR)
    LOGGER.info("Research charts written", extra={"dir": str(out), "count": len(figures)})
    return figures
