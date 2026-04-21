"""
core/result.py

Shared PipelineResult dataclass. Every module that produces or consumes
results must import from here — never define a parallel schema elsewhere.

FIX 7: Replaced datetime.utcnow() with datetime.now(tz=timezone.utc).
        utcnow() is deprecated in Python 3.12+ and returns a naive datetime
        with no timezone info, causing subtle comparison failures in cooldown logic.
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
        """width / height; returns 0 if height is zero."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def area_ratio(self) -> float:
        """Fraction of total image area occupied by this box."""
        return self.width * self.height

    def is_near_edge(self, proximity: float) -> bool:
        """Return True if box is within `proximity` of any image edge."""
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

    level: float          # 0–100 for water/food; ignored for mouse
    confidence: float     # 0–1
    label: str            # human-readable label from the model
    present: Optional[bool] = None  # used only for mouse target


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

    # Per-target confidences
    water_confidence: Optional[float] = None
    food_confidence: Optional[float] = None
    mouse_confidence: Optional[float] = None

    # Targets that were skipped due to gate rejection or detector miss
    uncertain_targets: List[str] = field(default_factory=list)

    # Full raw detection payload (for debugging / DB storage)
    raw_detections: Dict[str, Any] = field(default_factory=dict)

    # Where the image was saved after pipeline completion
    image_path: Optional[str] = None

    # FIX 7: timezone-aware datetime — utcnow() deprecated in Python 3.12+
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # DB-assigned identifier (populated after persistence)
    result_id: Optional[str] = None

    # Rejection reason when the image was rejected at preprocessor stage
    rejection_reason: Optional[str] = None

    # Whether the pipeline completed without a fatal error
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "result_id": self.result_id,
            "water_pct": self.water_pct,
            "food_pct": self.food_pct,
            "mouse_present": self.mouse_present,
            "water_confidence": self.water_confidence,
            "food_confidence": self.food_confidence,
            "mouse_confidence": self.mouse_confidence,
            "uncertain_targets": self.uncertain_targets,
            "raw_detections": self.raw_detections,
            "image_path": self.image_path,
            "timestamp": self.timestamp.isoformat(),
            "rejection_reason": self.rejection_reason,
            "success": self.success,
        }


@dataclass
class AlertRecord:
    """Represents a single fired alert (written to alert_log table)."""

    target: str
    alert_type: str          # low | missing | rejected
    value: Optional[float]
    message: str
    notifiers_fired: List[str] = field(default_factory=list)