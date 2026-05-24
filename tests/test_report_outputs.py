"""Report and dashboard output tests."""

from pathlib import Path

from src.analytics.charts.registry import build_chart_registry, write_standalone_charts
from src.analytics.unified_dashboard import write_unified_dashboard
from src.research.reporting.interpret import write_interpreted_report
from src.utils.config import load_app_config


def test_write_report_and_dashboard(tmp_path: Path) -> None:
    app = load_app_config()
    registry = build_chart_registry()
    figs = write_standalone_charts(registry, app, tmp_path / "research", tmp_path / "risk")
    dash_dir = tmp_path / "dashboard"
    if figs:
        write_unified_dashboard(dash_dir, figs, [s for s in registry if s.id in figs], app)
        assert (dash_dir / "index.html").is_file()
    report = tmp_path / "REPORT.md"
    write_interpreted_report(app, report)
    assert report.is_file()
    assert "Executive summary" in report.read_text(encoding="utf-8")
