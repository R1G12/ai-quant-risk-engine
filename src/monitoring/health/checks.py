"""Pipeline and artifact health checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.utils.config import AppConfig
from src.utils.paths import (
    RISK_DATASET_PATH,
    RISK_VAR_DIR,
    RESEARCH_BACKTESTS_DIR,
    RESEARCH_REPORT_MD_PATH,
)


@dataclass(frozen=True)
class HealthCheck:
    name: str
    ok: bool
    message: str


def _file_fresh(path: Path, warn_days: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"Missing: {path.name}"
    mtime = date.fromtimestamp(path.stat().st_mtime)
    age = (date.today() - mtime).days
    if age > warn_days:
        return True, f"Present but {age}d old (>{warn_days}d threshold)"
    return True, f"OK ({age}d old)"


def run_health_checks(app: AppConfig, *, freshness_warn_days: int = 30) -> list[HealthCheck]:
    """Check critical artifacts for platform readiness."""
    checks: list[HealthCheck] = []

    for label, path in [
        ("risk_dataset", RISK_DATASET_PATH),
        ("var_metrics", RISK_VAR_DIR / "var_metrics.parquet"),
        ("research_report", RESEARCH_REPORT_MD_PATH),
    ]:
        ok, msg = _file_fresh(path, freshness_warn_days)
        checks.append(HealthCheck(name=label, ok=ok, message=msg))

    bt = RESEARCH_BACKTESTS_DIR / f"experiment_id={app.research.meta.experiment_id}" / "equity_curve.parquet"
    ok, msg = _file_fresh(bt, freshness_warn_days)
    checks.append(HealthCheck(name="backtest_equity", ok=ok, message=msg))

    return checks
