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
from core.database import close_db, init_db
from core.job_tracker import JobTracker
from core.logger import setup_logging
from core.metrics import MetricsCollector
from core.task_queue import TaskQueue
from pipeline.orchestrator import PipelineOrchestrator


_orchestrator: PipelineOrchestrator | None = None
_task_queue: TaskQueue | None = None
_job_tracker: JobTracker | None = None
_metrics: MetricsCollector | None = None


def get_orchestrator() -> PipelineOrchestrator:
    if _orchestrator is None:
        raise RuntimeError("PipelineOrchestrator is not initialised")
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage pipeline startup and shutdown."""
    global _orchestrator, _task_queue, _job_tracker, _metrics

    config = get_config()
    setup_logging(config)

    logger.info(f"Starting {config.app.name} | env={config.app.env}")

    # Core singletons
    _orchestrator = PipelineOrchestrator(config)
    _job_tracker = JobTracker()
    _metrics = MetricsCollector()
    _task_queue = TaskQueue(config)

    await init_db(config)
    await _orchestrator.startup()
    _orchestrator.set_metrics(_metrics)
    await _task_queue.startup(_orchestrator, _job_tracker)

    # Expose to app state for dependency injection
    app.state.orchestrator = _orchestrator
    app.state.task_queue = _task_queue
    app.state.job_tracker = _job_tracker
    app.state.metrics = _metrics

    yield  # App is running here

    logger.info("Shutting down pipeline")
    await _task_queue.shutdown()
    await _orchestrator.shutdown()
    await close_db()
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

    # Routes
    from api.routes.ingest import router as ingest_router
    from api.routes.ingest_s3 import router as ingest_s3_router
    from api.routes.jobs import router as jobs_router
    from api.routes.metrics import router as metrics_router
    from api.routes.results import router as results_router

    app.include_router(ingest_router)
    app.include_router(ingest_s3_router)
    app.include_router(results_router)
    app.include_router(jobs_router)
    app.include_router(metrics_router)

    @app.get("/health", tags=["health"])
    async def health_check():
        return {
            "status": "ok",
            "env": config.app.env,
            "detector": config.detector.engine,
            "targets": list(config.targets.enabled),
        }

    return app


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
