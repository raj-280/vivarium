"""
api/routes/ingest.py

POST /analyze — receives an image upload, runs the full pipeline,
and returns a PipelineResult as JSON.

FIX: Removed duplicated _authenticate and _get_orchestrator.
     Both now imported from api.dependencies (single source of truth).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from loguru import logger

from api.dependencies import authenticate, get_orchestrator
from core.config_loader import get_config
from pipeline.orchestrator import PipelineOrchestrator

router = APIRouter(tags=["ingest"])


@router.post(
    "/analyze",
    summary="Analyze a vivarium image",
    response_description="Pipeline result with water, food, and mouse measurements",
    status_code=status.HTTP_200_OK,
)
async def analyze(
    request: Request,
    image: UploadFile = File(..., description="Image file to analyze"),
    _: str = Depends(authenticate),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
):
    """
    Accept a vivarium image, run the full monitoring pipeline, and return results.

    - **image**: multipart/form-data image file (jpg, jpeg, png, or webp)

    Returns the complete PipelineResult including water_pct, food_pct, mouse_present,
    confidence scores, and any uncertain targets.
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

    logger.info(f"POST /analyze | filename={filename} | bytes={len(image_bytes)}")

    try:
        result = await orchestrator.run(image_bytes, filename)
    except Exception as exc:
        logger.exception(f"Pipeline error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(exc)}",
        )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "image_rejected",
                "reason": result.rejection_reason,
                "result": result.to_dict(),
            },
        )

    output = result.to_dict()
    if not config.api.response_include_image_path:
        output.pop("image_path", None)

    return output