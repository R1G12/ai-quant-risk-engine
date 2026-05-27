"""Copilot Q&A inference endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter

from src.copilot.reasoning.engine import CopilotEngine

router = APIRouter()


class CopilotRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class CopilotAnswer(BaseModel):
    question: str
    title: str
    answer: str
    confidence: str
    requires_human_review: bool
    evidence: dict


@router.post("/copilot", response_model=CopilotAnswer)
def copilot_ask(body: CopilotRequest) -> CopilotAnswer:
    resp = CopilotEngine().ask(body.question)
    return CopilotAnswer(
        question=resp.question,
        title=resp.title,
        answer=resp.answer,
        confidence=resp.confidence,
        requires_human_review=resp.requires_human_review,
        evidence=resp.evidence,
    )
