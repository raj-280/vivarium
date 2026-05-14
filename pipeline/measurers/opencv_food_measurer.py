"""
pipeline/measurers/opencv_food_measurer.py

BaseMeasurer wrapper around the geometric FoodLevelDetector.
Registered in MeasurerFactory as engine key: 'opencv_food'

Mirrors the structure of opencv_water_measurer.py exactly — same pattern,
same interface, same config style. No model weights required; pure OpenCV.

Pipeline (delegated to FoodLevelDetector in food_level.py):
    1. Adaptive Canny edge detection (median-based thresholds)
    2. Hough transform → rough_surface_y  (dominant horizontal line)
    3. Reference-line regression → refined_surface_y  (peak edge-density row)
    4. surface_fusion: confidence-weighted average of both estimates
    5. food_pct = (bot_y - final_surface_y) / hopper_height × 100

Config block (config.yaml) — all fields are optional with sensible defaults:

    food:
      engine: opencv_food
      opencv_food:
        canny_low_frac: 0.33          # fraction of median intensity for Canny low
        canny_high_frac: 1.0          # fraction for Canny high
        hough_min_line_frac: 0.30     # min line length as fraction of ROI width
        hough_max_gap: 15             # max pixel gap in Hough line
        min_hough_lines: 1            # minimum lines to trust rough_surface_y
        surface_search_frac: 0.65     # top fraction of interior to search for surface
        ref_lines_y: [0.05, 0.95]     # known cage wall rows as fractions of height
        min_confidence: 0.20          # measurements below this are flagged uncertain
"""

from __future__ import annotations

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer
from pipeline.measurers.food_level import FoodLevelDetector


class OpenCVFoodMeasurer(BaseMeasurer):
    """
    Wraps FoodLevelDetector (geometric CV) into the BaseMeasurer interface.

    Can be used standalone (engine: opencv_food) or as the cv_engine inside
    ComparatorMeasurer for the 'food' target.

    No model weights required — all logic is pure OpenCV.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._detector: FoodLevelDetector | None = None
        self._min_confidence: float = 0.20

    def load(self) -> None:
        """
        Instantiate FoodLevelDetector from config.

        All constructor parameters are read from config.food.opencv_food.*
        If the opencv_food sub-block is absent, sensible defaults are used.
        No I/O or weight loading — this is instant.
        """
        # Read per-target config, then the opencv_food sub-block
        target_cfg = getattr(self.config.measurers, self.target, DotMap())
        cv_cfg = getattr(target_cfg, "opencv_food", DotMap())

        def _f(key: str, default: float) -> float:
            val = getattr(cv_cfg, key, None)
            return float(val) if val is not None else default

        def _i(key: str, default: int) -> int:
            val = getattr(cv_cfg, key, None)
            return int(val) if val is not None else default

        canny_low_frac      = _f("canny_low_frac",      0.33)
        canny_high_frac     = _f("canny_high_frac",     1.0)
        hough_min_line_frac = _f("hough_min_line_frac", 0.30)
        hough_max_gap       = _i("hough_max_gap",       15)
        min_hough_lines     = _i("min_hough_lines",     1)
        surface_search_frac = _f("surface_search_frac", 0.65)
        self._min_confidence = _f("min_confidence",     0.20)

        # ref_lines_y is a list — handle both DotMap list and Python list
        raw_ref = getattr(cv_cfg, "ref_lines_y", None)
        if raw_ref and not isinstance(raw_ref, DotMap):
            ref_lines_y = [float(v) for v in raw_ref]
        else:
            ref_lines_y = [0.05, 0.95]

        self._detector = FoodLevelDetector(
            canny_low_frac=canny_low_frac,
            canny_high_frac=canny_high_frac,
            hough_min_line_frac=hough_min_line_frac,
            hough_max_gap=hough_max_gap,
            min_hough_lines=min_hough_lines,
            ref_lines_y=ref_lines_y,
            surface_search_frac=surface_search_frac,
        )

        logger.info(
            f"[OpenCVFoodMeasurer] FoodLevelDetector ready | "
            f"canny=({canny_low_frac:.2f},{canny_high_frac:.2f}) "
            f"hough_min_line_frac={hough_min_line_frac} "
            f"surface_search_frac={surface_search_frac} "
            f"min_confidence={self._min_confidence}"
        )

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Run geometric food level estimation on the food hopper ROI.

        Args:
            roi: BGR numpy array — cropped food hopper region from the detector.

        Returns:
            MeasurementResult:
                level      — food fill % (0–100)
                confidence — fused confidence from Hough + regression (0–1)
                label      — human-readable breakdown string from FoodLevelDetector
                present    — None (not applicable for food level)

        Raises:
            RuntimeError: If .load() was not called before .measure().
        """
        if self._detector is None:
            raise RuntimeError(
                "OpenCVFoodMeasurer not loaded — call .load() first"
            )

        if roi.size == 0:
            logger.warning("[OpenCVFoodMeasurer] Empty ROI received — returning zero")
            return MeasurementResult(
                level=0.0,
                confidence=0.0,
                label="OpenCV food: empty ROI",
                present=None,
            )

        result = self._detector.detect(roi)

        # Flag low-confidence results in the label so the orchestrator
        # and downstream consumers can see the measurer was uncertain.
        if result.confidence < self._min_confidence:
            logger.warning(
                f"[OpenCVFoodMeasurer] Low confidence: {result.confidence:.3f} "
                f"< {self._min_confidence} | food_pct={result.food_pct:.1f}%"
            )
            label = (
                f"OpenCV food (LOW CONF {result.confidence:.2f}): "
                f"{result.food_pct:.1f}% — {result.label}"
            )
        else:
            label = result.label

        logger.debug(
            f"[OpenCVFoodMeasurer] food_pct={result.food_pct:.1f}% "
            f"conf={result.confidence:.3f} "
            f"rough_y={result.rough_surface_y}(c={result.rough_conf:.2f}) "
            f"refined_y={result.refined_surface_y}(c={result.refined_conf:.2f}) "
            f"final_y={result.final_surface_y}"
        )

        return MeasurementResult(
            level=result.food_pct,
            confidence=result.confidence,
            label=label,
            present=None,
        )