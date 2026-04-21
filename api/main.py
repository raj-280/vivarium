"""
api/main.py

FastAPI application factory. Creates the app, registers routes, and
manages the pipeline orchestrator lifecycle via startup/shutdown events.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config_loader import get_config
from core.logger import setup_logging
from pipeline.orchestrator import PipelineOrchestrator

# Module-level singleton — accessed by routes via app.state
_orchestrator: PipelineOrchestrator | None = None


def get_orchestrator() -> PipelineOrchestrator:
    """Dependency-injection helper for routes."""
    if _orchestrator is None:
        raise RuntimeError("PipelineOrchestrator is not initialised")
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage pipeline startup and shutdown."""
    global _orchestrator
    config = get_config()
    setup_logging(config)

    logger.info(f"Starting {config.app.name} | env={config.app.env}")
    _orchestrator = PipelineOrchestrator(config)
    await _orchestrator.startup()
    app.state.orchestrator = _orchestrator

    yield  # App is running here

    logger.info("Shutting down pipeline")
    await _orchestrator.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    config = get_config()

    app = FastAPI(
        title=config.app.name,
        version="1.0.0",
        description="Vivarium monitoring pipeline — water, food, and mouse detection",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from api.routes.ingest import router as ingest_router
    from api.routes.results import router as results_router

    app.include_router(ingest_router)
    app.include_router(results_router)

    @app.get("/health", tags=["health"])
    async def health_check():
        return {
            "status": "ok",
            "env": config.app.env,
            "detector": config.detector.engine,
            "targets": list(config.targets.enabled),
        }

    return app


# Create the application instance (imported by uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run(
        "api.main:app",
        host=cfg.api.host,
        port=int(cfg.api.port),
        reload=(cfg.app.env == "local"),
        log_level=cfg.app.log_level.lower(),
    )
