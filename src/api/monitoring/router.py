"""Health and observability endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from src.monitoring.alerts.rules import evaluate_alerts
from src.monitoring.health.checks import run_health_checks
from src.monitoring.quality.validators import run_quality_checks
from src.utils.config import load_app_config

router = APIRouter()


@router.get("/health")
def health() -> dict:
    app = load_app_config()
    checks = run_health_checks(app)
    return {
        "status": "ok" if all(c.ok for c in checks) else "degraded",
        "checks": [{"name": c.name, "ok": c.ok, "message": c.message} for c in checks],
    }


@router.get("/quality")
def quality() -> dict:
    app = load_app_config()
    checks = run_quality_checks(app)
    return {
        "checks": [{"name": c.name, "ok": c.ok, "message": c.message} for c in checks],
    }


@router.get("/alerts")
def alerts() -> dict:
    app = load_app_config()
    items = evaluate_alerts(app)
    return {"alerts": [{"severity": a.severity, "message": a.message} for a in items]}
