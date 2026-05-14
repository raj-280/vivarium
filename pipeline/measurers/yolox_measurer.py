"""
pipeline/measurers/yolox_measurer.py

YOLOXMeasurer — decodes level / condition from the class ID embedded in
the BoundingBox label by YOLOXDetector. No model weights, no inference.

Class ID → measurement mapping (matches yolox_vivarium_tiny.py):

    Water (classes 1–4):
        1 → Critical (7.5%)     2 → Low (25.0%)
        3 → OK (57.5%)          4 → Full (90.0%)

    Food (classes 5–8):
        5 → Critical (7.5%)     6 → Low (25.0%)
        7 → OK (57.5%)          8 → Full (90.0%)

    Mouse (class 0):
        present = True, level = 100.0

    Bedding (classes 9–12):
        9  → WORST    10 → BAD    11 → OK    12 → PERFECT

Engine key: yolox_class
"""

from __future__ import annotations

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer

# ── Water: class ID → fill percentage ────────────────────────────────────────
# Matches exp file: 1=critical(0-15%) 2=low(15-35%) 3=ok(35-80%) 4=full(80-100%)
_WATER_LEVEL: dict[int, float] = {
    1: 7.5,    # critical  — midpoint of 0–15%
    2: 25.0,   # low       — midpoint of 15–35%
    3: 57.5,   # ok        — midpoint of 35–80%
    4: 90.0,   # full      — midpoint of 80–100%
}
_WATER_LABEL: dict[int, str] = {
    1: "Critical",
    2: "Low",
    3: "OK",
    4: "Full",
}

# ── Food: class ID → fill percentage ─────────────────────────────────────────
# Matches exp file: 5=critical(0-15%) 6=low(15-35%) 7=ok(35-80%) 8=full(80-100%)
_FOOD_LEVEL: dict[int, float] = {
    5: 7.5,    # critical  — midpoint of 0–15%
    6: 25.0,   # low       — midpoint of 15–35%
    7: 57.5,   # ok        — midpoint of 35–80%
    8: 90.0,   # full      — midpoint of 80–100%
}
_FOOD_LABEL: dict[int, str] = {
    5: "Critical",
    6: "Low",
    7: "OK",
    8: "Full",
}

# ── Bedding: class ID → condition string ─────────────────────────────────────
# Matches exp file: 9=worst 10=bad 11=ok 12=perfect
_BEDDING_CONDITION: dict[int, str] = {
    9:  "WORST",
    10: "BAD",
    11: "OK",
    12: "PERFECT",
}


def _parse_cls_id(label: str) -> int:
    """
    Extract the integer class ID from a label like "water_cls2" or "bedding_cls11".
    Returns -1 if parsing fails.
    """
    try:
        return int(label.split("_cls")[-1])
    except (ValueError, IndexError):
        return -1


class YOLOXMeasurer(BaseMeasurer):
    """
    Passthrough measurer for YOLOX — decodes level from class ID in bbox label.

    No model weights. No inference. Instant.
    The BoundingBox.label produced by YOLOXDetector carries the class ID,
    e.g. "water_cls2". This measurer reads that, maps it to level/condition,
    and returns a MeasurementResult.

    Because the ROI crop is not used here, this measurer is fully compatible
    with the existing orchestrator loop — it just ignores the pixel data.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)

    def load(self) -> None:
        """Nothing to load — no model weights needed."""
        logger.info(
            f"[YOLOXMeasurer] Ready for target='{self.target}' "
            f"— level decoded from YOLOX class ID, no model needed"
        )

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Decode level / condition from the class ID stored in self._last_label.

        Note: The orchestrator passes the pixel ROI here, but we don't use it.
        The class ID was already embedded in the BoundingBox.label by YOLOXDetector
        and is injected via set_last_label() before measure() is called.

        Returns:
            MeasurementResult with level, confidence, label, and bedding fields
            populated as appropriate for the target.
        """
        label = getattr(self, "_last_label", None)
        conf  = getattr(self, "_last_confidence", 1.0)

        if label is None:
            logger.warning(
                f"[YOLOXMeasurer] target='{self.target}' — "
                f"no label set, returning unknown result"
            )
            return MeasurementResult(
                level=0.0,
                confidence=0.0,
                label="NOT_DETECTED",
                present=False if self.target == "mouse" else None,
                bedding_condition="NOT_DETECTED" if self.target == "bedding" else None,
            )

        cls_id = _parse_cls_id(label)

        # ── Mouse ─────────────────────────────────────────────────────
        if self.target == "mouse":
            logger.debug(f"[YOLOXMeasurer] mouse detected | cls_id={cls_id}")
            return MeasurementResult(
                level=100.0,
                confidence=conf,
                label="mouse detected",
                present=True,
            )

        # ── Water ─────────────────────────────────────────────────────
        if self.target == "water":
            level = _WATER_LEVEL.get(cls_id, 0.0)
            name  = _WATER_LABEL.get(cls_id, "Unknown")
            result_label = f"water {name} (cls{cls_id})"
            logger.debug(f"[YOLOXMeasurer] water | cls_id={cls_id} → {level}%")
            return MeasurementResult(
                level=level,
                confidence=conf,
                label=result_label,
            )

        # ── Food ──────────────────────────────────────────────────────
        if self.target == "food":
            level = _FOOD_LEVEL.get(cls_id, 0.0)
            name  = _FOOD_LABEL.get(cls_id, "Unknown")
            result_label = f"food {name} (cls{cls_id})"
            logger.debug(f"[YOLOXMeasurer] food | cls_id={cls_id} → {level}%")
            return MeasurementResult(
                level=level,
                confidence=conf,
                label=result_label,
            )

        # ── Bedding ───────────────────────────────────────────────────
        if self.target == "bedding":
            condition = _BEDDING_CONDITION.get(cls_id, "NOT_DETECTED")
            result_label = f"bedding {condition} (cls{cls_id})"
            logger.debug(f"[YOLOXMeasurer] bedding | cls_id={cls_id} → {condition}")
            return MeasurementResult(
                level=0.0,
                confidence=conf,
                label=result_label,
                bedding_condition=condition,
                bedding_area_pct=getattr(self, "_last_area_pct", None),
            )

        # ── Unknown target ────────────────────────────────────────────
        logger.warning(f"[YOLOXMeasurer] unknown target='{self.target}'")
        return MeasurementResult(level=0.0, confidence=0.0, label="UNKNOWN")

    def set_last_label(self, label: str, area_pct: float = 0.0, confidence: float = 1.0) -> None:
        """
        Called by the orchestrator BEFORE measure() to pass the bbox label.

        The orchestrator injects the BoundingBox.label here so the measurer
        can decode the class ID without needing access to the bbox object itself.

        Args:
            label:      BoundingBox.label from YOLOXDetector, e.g. "water_cls2"
            area_pct:   bbox area as fraction of frame (width * height), for bedding
            confidence: raw detector confidence score to pass through to result
        """
        self._last_label = label
        self._last_area_pct = area_pct
        self._last_confidence = confidence
