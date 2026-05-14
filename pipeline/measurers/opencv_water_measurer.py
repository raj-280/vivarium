"""
pipeline/measurers/opencv_water_measurer.py

BaseMeasurer wrapper around the geometric WaterLevelDetector.
Registered in MeasurerFactory as engine key: 'opencv_water'

Phase 1 Step 3: MISSING bottle detection.
  Before running the fill-line estimator we check two signals that indicate
  the bottle is simply not there (as opposed to being empty):

    1. Detection bbox area is very small  (< missing_area_threshold fraction
       of the full image) — the YOLO box is probably a false positive on a
       cage bar or a tiny reflection.

    2. The ROI is nearly uniform brightness — a real bottle has glass
       edges and a liquid meniscus; a plain cage wall has almost no variance.
       If std-dev of the grayscale ROI < missing_variance_threshold the
       region is considered featureless → MISSING.

  When either signal fires the measurer returns:
    level      = 0.0   (so level_labeler won't crash)
    confidence = 0.0
    label      = "MISSING"
    present    = False

Config keys (under measurers.water):
  missing_area_threshold:     0.005   bbox area fraction below which → MISSING
  missing_variance_threshold: 8.0     ROI grayscale std-dev below which → MISSING
  min_confidence:             0.5     passed through to UNDETERMINED gate
"""

from __future__ import annotations

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer
from pipeline.measurers.water_level import WaterLevelDetector

_MISSING = "MISSING"


class OpenCVWaterMeasurer(BaseMeasurer):
    """
    Wraps WaterLevelDetector (geometric CV) into the BaseMeasurer interface.
    Adds MISSING bottle detection before running fill-line estimation.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._detector: WaterLevelDetector | None = None

        target_cfg = getattr(config.measurers, target, DotMap())
        self._missing_area_thresh: float = float(
            getattr(target_cfg, "missing_area_threshold", 0.005)
        )
        self._missing_var_thresh: float = float(
            getattr(target_cfg, "missing_variance_threshold", 8.0)
        )

    def load(self) -> None:
        """Instantiate the geometric detector (no model weights needed)."""
        self._detector = WaterLevelDetector()
        logger.info(
            f"[OpenCVWaterMeasurer] WaterLevelDetector ready | "
            f"missing_area_thresh={self._missing_area_thresh} "
            f"missing_var_thresh={self._missing_var_thresh}"
        )

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        if self._detector is None:
            raise RuntimeError("OpenCVWaterMeasurer not loaded — call .load() first")

        # ── MISSING check ─────────────────────────────────────────────────
        missing_reason = self._check_missing(roi)
        if missing_reason:
            logger.info(f"[OpenCVWaterMeasurer] Bottle MISSING — {missing_reason}")
            return MeasurementResult(
                level=0.0,
                confidence=0.0,
                label=_MISSING,
                present=False,
            )

        # ── Normal fill-line estimation ───────────────────────────────────
        result = self._detector.detect(roi)

        return MeasurementResult(
            level=result.water_pct,
            confidence=result.confidence,
            label=result.label,
            present=None,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _check_missing(self, roi: np.ndarray) -> str:
        """
        Return a non-empty reason string if the bottle appears to be missing,
        or an empty string if the bottle looks real.

        Checks (in order):
          1. ROI too small  → likely a false-positive bbox on a cage bar
          2. ROI too uniform → featureless region, no glass/liquid texture
        """
        h, w = roi.shape[:2]
        roi_pixels = h * w

        # Check 1 — absolute minimum size (avoids division errors too)
        if roi_pixels < 100:
            return f"ROI too small ({roi_pixels}px)"

        # Check 2 — variance / texture check
        gray = roi if roi.ndim == 2 else _to_gray(roi)
        std_dev = float(np.std(gray.astype(np.float32)))
        if std_dev < self._missing_var_thresh:
            return (
                f"ROI too uniform (std={std_dev:.1f} < "
                f"threshold={self._missing_var_thresh})"
            )

        return ""  # bottle present, proceed with estimation


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale using OpenCV (imported lazily to keep top-level clean)."""
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
