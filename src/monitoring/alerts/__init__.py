"""Alert hooks (extensible)."""

from src.monitoring.alerts.rules import Alert, evaluate_alerts

__all__ = ["Alert", "evaluate_alerts"]
