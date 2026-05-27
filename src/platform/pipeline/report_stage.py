"""DVC stage: generate_platform_reports."""

from __future__ import annotations

from src.platform.reporting.generator import generate_all_reports
from src.utils.config import load_app_config
from src.utils.logger import get_logger
from src.utils.metrics import log_stage_metrics
from src.utils.paths import METRICS_DIR, PLATFORM_METRICS_DIR, ensure_dir

LOGGER = get_logger(__name__)


def run() -> None:
    app = load_app_config()
    paths = generate_all_reports(app)
    ensure_dir(PLATFORM_METRICS_DIR)
    log_stage_metrics(
        PLATFORM_METRICS_DIR,
        {"reports_generated": len(paths)},
    )
    LOGGER.info("Platform reports stage complete", extra={"count": len(paths)})


if __name__ == "__main__":
    run()
