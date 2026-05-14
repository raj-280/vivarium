"""
core/result.py

Shared PipelineResult dataclass. Every module that produces or consumes
results must import from here — never define a parallel schema elsewhere.

FIX 7: Replaced datetime.utcnow() with datetime.now(tz=timezone.utc).
        utcnow() is deprecated in Python 3.12+ and returns a naive datetime
        with no timezone info.

YOLOX update: Added bedding fields to MeasurementResult and PipelineResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class BoundingBox:
    """Normalised bounding box in [x1, y1, x2, y2] format (0–1 scale)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    def is_near_edge(self, proximity: float) -> bool:
        return (
            self.x1 < proximity
            or self.y1 < proximity
            or self.x2 > (1.0 - proximity)
            or self.y2 > (1.0 - proximity)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "label": self.label,
        }


@dataclass
class MeasurementResult:
    """Output produced by a single measurer for one target."""

    level: float                              # 0–100 for water/food; 0 for mouse/bedding
    confidence: float                         # 0–1
    label: str                                # human-readable label from the model
    present: Optional[bool] = None            # used only for mouse target
    # Bedding-specific — only populated when target == "bedding"
    bedding_condition: Optional[str] = None   # PERFECT | OK | BAD | WORST | NOT_DETECTED
    bedding_area_pct: Optional[float] = None  # bbox area as fraction of frame (0–1)


@dataclass
class PipelineResult:
    """
    Unified output of the vivarium monitoring pipeline.
    All fields that may be absent are Optional with a sensible default.
    """

    # Core measurements
    water_pct: Optional[float] = None
    food_pct: Optional[float] = None
    mouse_present: Optional[bool] = None

    # Human-readable labels for water and food levels
    water_label: Optional[str] = None
    food_label: Optional[str] = None

    # Per-target confidences
    water_confidence: Optional[float] = None
    food_confidence: Optional[float] = None
    mouse_confidence: Optional[float] = None

    # Bedding fields — populated when "bedding" is in targets.enabled
    bedding_condition: Optional[str] = None    # PERFECT | OK | BAD | WORST | NOT_DETECTED
    bedding_confidence: Optional[float] = None
    bedding_area_pct: Optional[float] = None   # bbox area as fraction of frame (0–1)

    # Targets skipped due to gate rejection or detector miss
    uncertain_targets: List[str] = field(default_factory=list)

    # Full raw detection payload (for debugging / DB storage)
    raw_detections: Dict[str, Any] = field(default_factory=dict)

    # Where the annotated image was saved
    image_path: Optional[str] = None

    # Mouse stationary flag
    mouse_stationary: Optional[bool] = None

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    result_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    success: bool = True

    def to_dict(self, cage_id: Optional[str] = None) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict — returned to API, webhook, and logs."""
        return {
            "cage_id": cage_id,
            "water_pct": self.water_pct,
            "water_label": self.water_label,
            "food_pct": self.food_pct,
            "food_label": self.food_label,
            "mouse_present": self.mouse_present,
            "mouse_stationary": self.mouse_stationary,
            "bedding_condition": self.bedding_condition,
            "bedding_confidence": self.bedding_confidence,
            "bedding_area_pct": self.bedding_area_pct,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "image_path": self.image_path,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AlertRecord:
    """Represents a single fired alert (written to alert_log table)."""

    target: str
    alert_type: str        # LOW | EMPTY
    value: Optional[float]
    message: str
