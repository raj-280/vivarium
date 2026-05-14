"""
api/routes/ingest.py

POST /analyze — receives an image upload, enqueues it for async processing,
and returns a request_id immediately (non-blocking).

Poll GET /job/{request_id} for status and result.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from loguru import logger

from api.dependencies import get_job_tracker, get_metrics, get_orchestrator, get_task_queue
from core.config_loader import get_config
from pipeline.orchestrator import PipelineOrchestrator

router = APIRouter(tags=["ingest"])


@router.post(
    "/analyze",
    summary="Submit a vivarium image for async analysis",
    response_description="request_id to poll via GET /job/{request_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze(
    request: Request,
    cage_id: str = Query(..., description="Unique identifier for the cage (e.g. cage_1)"),
    image: UploadFile = File(..., description="Image file to analyze (jpg, jpeg, png, webp)"),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
):
    """
    Submit an image for async pipeline processing.

    Returns a `request_id` immediately. Poll `GET /job/{request_id}` for status.

    Status values: `PENDING` → `PROCESSING` → `DONE` | `FAILED`
    """
    config = get_config()

    # Filename / format validation
    allowed_formats = set(config.input.allowed_formats)
    filename = image.filename or "upload.jpg"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File extension '{ext}' not allowed. Allowed: {list(allowed_formats)}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file received",
        )

    logger.info(f"POST /analyze | cage={cage_id} | filename={filename} | bytes={len(image_bytes)}")

    task_queue = get_task_queue()
    job_tracker = get_job_tracker()
    metrics = get_metrics()

    # Enqueue — returns request_id immediately
    try:
        request_id = await task_queue.enqueue(image_bytes, filename, cage_id)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue is full. Try again shortly.",
        )

    # Register as PENDING in tracker
    job_tracker.set_pending(request_id, cage_id)

    # Update queue backlog metric
    metrics.set_queue_backlog(task_queue.qsize())

    return {
        "request_id": request_id,
        "status": "PENDING",
        "message": f"Job queued. Poll GET /job/{request_id} for result.",
    }
