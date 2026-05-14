"""
api/routes/ingest_s3.py

POST /analyze/s3 — accepts an S3 URI or bucket+key, fetches the image,
enqueues it for async processing, and returns a request_id immediately
(non-blocking) — identical lifecycle to POST /analyze.

Request body (JSON):
    {
      "cage_id":  "cage_1",
      "s3_uri":   "s3://my-bucket/cage_1/frame_001.jpg"   // option A
    }
    — OR —
    {
      "cage_id": "cage_1",
      "bucket":  "my-bucket",                             // option B
      "key":     "cage_1/frame_001.jpg"
    }

Poll GET /job/{request_id} for status and result — same as POST /analyze.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, model_validator
from typing import Optional

from api.dependencies import get_job_tracker, get_metrics, get_task_queue
from core.config_loader import get_config
from core.s3_fetcher import S3FetchError, S3ImageFetcher

router = APIRouter(tags=["ingest"])


# ── Request schema ────────────────────────────────────────────────────────────

class S3AnalyzeRequest(BaseModel):
    cage_id: str
    s3_uri: Optional[str] = None    # e.g. "s3://my-bucket/cage_1/frame.jpg"
    bucket: Optional[str] = None    # explicit bucket (alternative to s3_uri)
    key: Optional[str] = None       # explicit key   (alternative to s3_uri)

    @model_validator(mode="after")
    def check_source(self) -> "S3AnalyzeRequest":
        if not self.s3_uri and not self.key:
            raise ValueError("Provide either 's3_uri' or 'key' (with optional 'bucket')")
        return self


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze/s3",
    summary="Submit a vivarium image from S3 for async analysis",
    response_description="request_id to poll via GET /job/{request_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_s3(
    request: Request,
    body: S3AnalyzeRequest,
):
    """
    Fetch an image from AWS S3 and enqueue it for async pipeline processing.

    Supply either a full **s3_uri** (`s3://bucket/key`) or a **bucket** +
    **key** pair. The cage_id is always required.

    Returns a `request_id` immediately. Poll `GET /job/{request_id}` for status.

    Status values: `PENDING` → `PROCESSING` → `DONE` | `FAILED`
    """
    config = get_config()
    fetcher = S3ImageFetcher(config)

    # ── Fetch from S3 ─────────────────────────────────────────────────────
    try:
        image_bytes, filename = fetcher.fetch(
            s3_uri=body.s3_uri,
            bucket=body.bucket,
            key=body.key,
        )
    except S3FetchError as exc:
        logger.warning(f"S3 fetch failed | cage={body.cage_id} | {exc.reason}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "s3_fetch_failed", "reason": exc.reason},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_request", "reason": str(exc)},
        )

    # ── Format check ──────────────────────────────────────────────────────
    allowed_formats = set(config.input.allowed_formats)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File extension '{ext}' not allowed. Allowed: {list(allowed_formats)}",
        )

    logger.info(
        f"POST /analyze/s3 | cage={body.cage_id} | filename={filename} "
        f"| bytes={len(image_bytes)}"
    )

    task_queue = get_task_queue()
    job_tracker = get_job_tracker()
    metrics = get_metrics()

    # ── Enqueue — returns request_id immediately ───────────────────────────
    try:
        request_id = await task_queue.enqueue(image_bytes, filename, body.cage_id)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue is full. Try again shortly.",
        )

    # Register as PENDING in tracker
    job_tracker.set_pending(request_id, body.cage_id)

    # Update queue backlog metric
    metrics.set_queue_backlog(task_queue.qsize())

    return {
        "request_id": request_id,
        "status": "PENDING",
        "message": f"Job queued. Poll GET /job/{request_id} for result.",
    }
