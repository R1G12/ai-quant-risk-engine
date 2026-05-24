"""Chart specification base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.utils.config import AppConfig


@dataclass
class ChartSpec:
    """Metadata + builder for a single dashboard chart."""

    id: str
    title: str
    description: str
    group: str
    builder: Callable[[AppConfig], object | None]
    standalone_name: str | None = None
