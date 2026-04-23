"""
pipeline/measurers/opencv_food_measurer.py

OpenCV food-level measurer for top-down vivarium bowl images.
Uses HSV color segmentation to separate food pellets from empty bowl area.
Much more reliable than CLIP for estimating fill level from above.

Config block required under food: in config.yaml:

    food:
      engine: opencv_food
      opencv_food:
        # HSV range for food pellets (dark brown/tan seeds)
        # Tune these with the debug flag if your pellets look different
        pellet_h_low:  5
        pellet_h_high: 30
        pellet_s_low:  30
        pellet_s_high: 255
        pellet_v_low:  20
        pellet_v_high: 180

        # Minimum edge pixels before we trust the measurement
        min_edge_pixels: 50

        # Debug: saves masked image to outputs/debug/ if true
        debug: false

Level mapping (fill ratio → reported percentage):
    >= 75% covered  → 100
    >= 45% covered  → 75
    >= 25% covered  → 50
    >= 10% covered  → 25
    <  10% covered  → 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


# Fill ratio → reported level.  Ordered highest → lowest.
_LEVEL_THRESHOLDS = [
    (0.75, 100.0),
    (0.45, 75.0),
    (0.25, 50.0),
    (0.10, 25.0),
    (0.00,  0.0),
]


class OpenCVFoodMeasurer(BaseMeasurer):
    """
    Top-down food level estimator using HSV pellet segmentation.

    Works by:
      1. Converting the bowl ROI to HSV colour space.
      2. Masking pixels that fall within the pellet HSV range.
      3. Computing fill_ratio = pellet_pixels / total_bowl_pixels.
      4. Mapping fill_ratio to a discrete level via _LEVEL_THRESHOLDS.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)

    def load(self) -> None:
        """No model to load — just verify cv2 is importable."""
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "opencv-python is not installed. Run: pip install opencv-python"
            ) from exc
        logger.info(f"OpenCVFoodMeasurer ready for target '{self.target}'")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Estimate food fill level from a top-down bowl crop.

        Args:
            roi: BGR numpy array — cropped bowl region from the preprocessed frame.

        Returns:
            MeasurementResult with level (0/25/50/75/100), confidence, label.
        """
        import cv2  # type: ignore

        target_cfg = getattr(self.config, self.target)
        cfg = target_cfg.opencv_food

        # --- 1. Read HSV bounds from config ---
        h_low  = int(getattr(cfg, "pellet_h_low",  5))
        h_high = int(getattr(cfg, "pellet_h_high", 30))
        s_low  = int(getattr(cfg, "pellet_s_low",  30))
        s_high = int(getattr(cfg, "pellet_s_high", 255))
        v_low  = int(getattr(cfg, "pellet_v_low",  20))
        v_high = int(getattr(cfg, "pellet_v_high", 180))
        min_edge_px = int(getattr(cfg, "min_edge_pixels", 50))
        debug = bool(getattr(cfg, "debug", False))

        lower = np.array([h_low,  s_low,  v_low],  dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        # --- 2. Convert to HSV and mask pellet pixels ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Light morphological cleanup — remove salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- 3. Compute fill ratio ---
        total_pixels  = roi.shape[0] * roi.shape[1]
        pellet_pixels = int(np.count_nonzero(mask))
        fill_ratio    = pellet_pixels / max(total_pixels, 1)

        # --- 4. Confidence: trust the reading more when more pellet px found ---
        # Low pellet count → unreliable (empty bowl or wrong crop)
        if pellet_pixels < min_edge_px:
            logger.warning(
                f"OpenCVFood [{self.target}] LOW SIGNAL: "
                f"pellet_pixels={pellet_pixels} < {min_edge_px} — "
                f"returning uncertain"
            )
            return MeasurementResult(
                level=None,
                confidence=float(fill_ratio),
                label=f"Low signal ({pellet_pixels} pellet px) — measurement uncertain",
                present=None,
            )

        confidence = float(min(pellet_pixels / max(total_pixels * 0.10, 1), 1.0))

        # --- 5. Map fill ratio to discrete level ---
        level = 0.0
        for threshold, mapped_level in _LEVEL_THRESHOLDS:
            if fill_ratio >= threshold:
                level = mapped_level
                break

        label = (
            f"Food level {level:.0f}% "
            f"(pellet coverage {fill_ratio * 100:.1f}%)"
        )

        logger.debug(
            f"OpenCVFood [{self.target}] → fill_ratio={fill_ratio:.3f} "
            f"pellet_px={pellet_pixels} total_px={total_pixels} "
            f"level={level} conf={confidence:.3f}"
        )

        # --- 6. Optional debug: save masked image ---
        if debug:
            try:
                debug_dir = Path("outputs/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                import time
                ts = int(time.time())
                cv2.imwrite(str(debug_dir / f"food_mask_{ts}.jpg"), mask)
                cv2.imwrite(str(debug_dir / f"food_roi_{ts}.jpg"),  roi)
                logger.debug(f"OpenCVFood debug images saved → {debug_dir}")
            except Exception as exc:
                logger.debug(f"Debug save failed (non-fatal): {exc}")

        return MeasurementResult(
            level=level,
            confidence=confidence,
            label=label,
            present=None,
        )