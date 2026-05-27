"""Tests for institutional report generation."""

from __future__ import annotations

from pathlib import Path

from src.platform.reporting.generator import generate_all_reports
from src.utils.config import load_app_config


def test_generate_all_reports_writes_files(tmp_path, monkeypatch) -> None:
    from src.utils import paths as p

    monkeypatch.setattr(p, "REPORTS_PORTFOLIO_DIR", tmp_path / "portfolio")
    monkeypatch.setattr(p, "REPORTS_RISK_DIR", tmp_path / "risk")
    monkeypatch.setattr(p, "REPORTS_GOVERNANCE_DIR", tmp_path / "governance")
    monkeypatch.setattr(p, "REPORTS_SIMULATIONS_DIR", tmp_path / "simulations")
    app = load_app_config()
    out = generate_all_reports(app)
    assert "portfolio" in out
    assert out["portfolio"].is_file()
    assert out["governance"].is_file()
