"""
api/schemas/ingest.py

Pydantic response model for POST /analyze and POST /analyze/s3.
Defines the exact shape of the API response.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    cage_id: Optional[str] = None
    water_pct: Optional[float] = None
    water_label: Optional[str] = None
    food_pct: Optional[float] = None
    food_label: Optional[str] = None
    mouse_present: Optional[bool] = None
    mouse_stationary: Optional[bool] = None
    timestamp: str
    success: bool
    image_path: Optional[str] = None
    rejection_reason: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "cage_id": "cage_1",
            "water_pct": 72.5,
            "water_label": "OK",
            "food_pct": 45.0,
            "food_label": "Low",
            "mouse_present": True,
            "mouse_stationary": False,
            "timestamp": "2026-05-11T10:00:00Z",
            "success": True,
            "image_path": "outputs/annotated/cage_1_frame.jpg",
            "rejection_reason": None,
        }
    }}