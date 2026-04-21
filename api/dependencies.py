"""
api/dependencies.py

Shared FastAPI dependencies — authentication, orchestrator access.
Import from here in all routes. Never duplicate dependency logic.

FIX: _authenticate was duplicated in ingest.py and results.py.
     Moved here so any auth change is made in exactly one place.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from core.config_loader import get_config
from pipeline.orchestrator import PipelineOrchestrator


def get_orchestrator(request: Request) -> PipelineOrchestrator:
    """Extract the pipeline orchestrator from app state (set during startup)."""
    return request.app.state.orchestrator


def authenticate(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Validate the X-API-Key header against config.api.api_key.

    Raises:
        HTTPException 401: If the key is missing or does not match.
    """
    config = get_config()
    expected_key: str = config.api.api_key
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key