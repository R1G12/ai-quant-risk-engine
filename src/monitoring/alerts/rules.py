"""Simple threshold alerts from health and drift checks."""

from __future__ import annotations

from dataclasses import dataclass

from src.monitoring.drift.baseline import compute_feature_drift
from src.monitoring.health.checks import run_health_checks
from src.utils.config import AppConfig


@dataclass(frozen=True)
class Alert:
    severity: str
    message: str


def evaluate_alerts(app: AppConfig) -> list[Alert]:
    alerts: list[Alert] = []
    for check in run_health_checks(app):
        if not check.ok:
            alerts.append(Alert("critical", f"{check.name}: {check.message}"))
    for drift in compute_feature_drift():
        if drift.flagged:
            alerts.append(
                Alert(
                    "warning",
                    f"Drift on {drift.feature}: z={drift.zscore:.2f}",
                )
            )
    return alerts
