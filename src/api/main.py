"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.monitoring.router import router as monitoring_router
from src.api.portfolio.router import router as portfolio_router
from src.api.risk.router import router as risk_router
from src.api.inference.router import router as inference_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Quant Risk Engine API",
        version="0.5.0",
        description="Portfolio intelligence, risk metrics, and copilot inference.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])
    app.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
    app.include_router(risk_router, prefix="/risk", tags=["risk"])
    app.include_router(inference_router, prefix="/inference", tags=["inference"])
    return app


app = create_app()
