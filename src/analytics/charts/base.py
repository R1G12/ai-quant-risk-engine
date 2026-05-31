"""Chart specification base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.analytics.charts.context import ChartContext


@dataclass
class ChartSpec:
    """Metadata + builder for a single dashboard chart."""

    id: str
    title: str
    description: str
    group: str
    builder: Callable[[ChartContext], object | None]
    standalone_name: str | None = None
    date_filterable: bool = False
