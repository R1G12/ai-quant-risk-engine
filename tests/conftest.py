"""Pytest hooks — enforce Python 3.12 (matches CI and .venv312)."""

from __future__ import annotations

import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 12):
        pytest.exit(
            f"This project requires Python 3.12.x (use .venv312). "
            f"Current interpreter: {sys.version}",
            returncode=1,
        )
