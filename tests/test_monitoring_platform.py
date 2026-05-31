"""Tests for Phase 5 monitoring."""

from __future__ import annotations

from src.monitoring.health.checks import run_health_checks
from src.monitoring.quality.validators import run_quality_checks
from src.monitoring.alerts.rules import evaluate_alerts
from src.utils.config import load_app_config


def test_run_health_checks_returns_list() -> None:
    app = load_app_config()
    checks = run_health_checks(app)
    assert len(checks) >= 3
    assert all(c.name for c in checks)


def test_run_quality_checks_with_dataset() -> None:
    app = load_app_config()
    checks = run_quality_checks(app)
    assert any(c.name == "risk_dataset_rows" for c in checks)


def test_evaluate_alerts_runs() -> None:
    alerts = evaluate_alerts(load_app_config())
    assert isinstance(alerts, list)
