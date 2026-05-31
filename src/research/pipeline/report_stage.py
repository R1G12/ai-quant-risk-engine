"""DVC: generate_research_reports."""

from __future__ import annotations

import json

from src.analytics.charts.registry import build_chart_registry, write_standalone_charts
from src.analytics.unified_dashboard import write_unified_dashboard
from src.research.reporting.interpret import write_interpreted_report
from src.utils.config import load_app_config
from src.utils.experiment import log_research_metrics
from src.utils.logger import get_logger
from src.utils.paths import (
    ANALYTICS_DASHBOARD_DIR,
    ANALYTICS_RESEARCH_DIR,
    ANALYTICS_RISK_DIR,
    METRICS_DIR,
    RESEARCH_PERFORMANCE_PATH,
    RESEARCH_REPORT_MD_PATH,
    RESEARCH_REPORTS_DIR,
    ensure_dir,
)

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    exp_id = app.research.meta.experiment_id
    ensure_dir(ANALYTICS_RESEARCH_DIR)
    ensure_dir(ANALYTICS_RISK_DIR)
    ensure_dir(RESEARCH_REPORTS_DIR)

    registry = build_chart_registry()
    all_figs = write_standalone_charts(registry, app, ANALYTICS_RESEARCH_DIR, ANALYTICS_RISK_DIR)
    write_unified_dashboard(ANALYTICS_DASHBOARD_DIR, all_figs, registry, app)
    write_interpreted_report(app, RESEARCH_REPORT_MD_PATH)

    summary = {
        "experiment_id": exp_id,
        "performance_summary": str(RESEARCH_PERFORMANCE_PATH),
        "analytics_dir": str(ANALYTICS_RESEARCH_DIR),
        "dashboard": str(ANALYTICS_DASHBOARD_DIR / "index.html"),
        "report_md": str(RESEARCH_REPORT_MD_PATH),
    }
    meta_path = RESEARCH_REPORTS_DIR / "_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_research_metrics(
        METRICS_DIR / "research_reports",
        {"reports_generated": 1, "n_charts": len(all_figs)},
        experiment_id=exp_id,
        stage="generate_research_reports",
        outputs=[str(ANALYTICS_DASHBOARD_DIR), str(RESEARCH_REPORT_MD_PATH)],
        enable_dvc_exp=app.research.meta.enable_dvc_experiments,
    )
    LOGGER.info("Research reports generated", extra={"experiment_id": exp_id, "charts": len(all_figs)})


if __name__ == "__main__":
    run()
