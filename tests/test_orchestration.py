"""Tests for orchestration workflows."""

from __future__ import annotations

import pytest

from src.orchestration.scheduler import run_workflow
from src.orchestration.workflows import WORKFLOWS


def test_workflows_defined() -> None:
    assert "refresh_market" in WORKFLOWS
    assert "platform_reports" in WORKFLOWS
    assert WORKFLOWS["platform_reports"].dvc_stages == ["generate_platform_reports"]


def test_run_workflow_dry_run(capsys) -> None:
    code = run_workflow("refresh_market", dry_run=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "dvc repro" in out
    assert "ingest_market_data" in out


def test_run_workflow_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown workflow"):
        run_workflow("not_a_workflow", dry_run=True)
