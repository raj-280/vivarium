"""
api/errors.py

Centralized error handling.

Rules:
  - NEVER send raw str(exc) or tracebacks to the client.
  - Always log the full exception server-side with logger.exception().
  - Always return a short error_id the client can quote in a support request.
  - Keep error surface small: error type + error_id only.
"""

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def make_error_id() -> str:
    """Generate a short 8-char correlation ID for support tracing."""
    return str(uuid.uuid4())[:8]


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception that reaches the ASGI layer.
    Logs full traceback server-side. Returns only a safe error_id to the client.
    """
    error_id = make_error_id()
    logger.exception(
        f"Unhandled error | error_id={error_id} | "
        f"method={request.method} | path={request.url.path}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "error_id": error_id,
            "detail": "An unexpected error occurred. Quote the error_id when reporting this.",
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic / FastAPI request validation errors.
    Returns field-level detail so clients know exactly what was wrong.
    Does NOT log as an exception — bad input is not a server error.
    """
    logger.warning(
        f"Validation error | path={request.url.path} | errors={exc.errors()}"
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
        },
    )