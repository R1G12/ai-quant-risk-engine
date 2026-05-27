"""Inference orchestration with simple in-memory cache."""

from __future__ import annotations

from functools import lru_cache

from src.copilot.reasoning.engine import CopilotEngine, CopilotResponse


@lru_cache(maxsize=128)
def cached_ask(question: str) -> CopilotResponse:
    return CopilotEngine().ask(question)
