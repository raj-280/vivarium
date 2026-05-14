"""
pipeline/measurers/detection_presence_measurer.py

A zero-model measurer for targets where presence is determined purely
by whether the detector found a bounding box.

Used for 'mouse' target — YOLOv8 already answers "is the mouse present?"
via its bounding box + confidence score. Running a second classifier on
top is redundant. This measurer simply returns present=True since if this
measurer is called at all, the detector already found the target.

Engine key: detection_presence
Config block required: none — no model path, no labels needed.

  mouse:
    engine: detection_presence
    min_confidence: 0.45   # this is used by the detector, not this measurer
"""

from __future__ import annotations

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


class DetectionPresenceMeasurer(BaseMeasurer):
    """
    Presence measurer that trusts the detector output directly.

    If the orchestrator calls measure(), it means the detector already
    found a bounding box for this target with sufficient confidence.
    So present=True unconditionally, and we use the detector's confidence
    score passed via the bbox (stored in config at runtime).

    No model weights. No inference. Instant.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)

    def load(self) -> None:
        """Nothing to load — no model weights needed."""
        logger.info(
            f"[DetectionPresenceMeasurer] Ready for target '{self.target}' "
            f"— presence derived directly from detector output"
        )

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Return present=True — detector already confirmed this target exists.

        Args:
            roi: Cropped BGR numpy array (not used, but required by interface).

        Returns:
            MeasurementResult with present=True, level=100, confidence=1.0.
        """
        logger.debug(
            f"[DetectionPresenceMeasurer] target='{self.target}' "
            f"→ present=True (detector confirmed)"
        )
        return MeasurementResult(
            level=100.0,
            confidence=1.0,
            label=f"{self.target} detected by detector",
            present=True,
        )
