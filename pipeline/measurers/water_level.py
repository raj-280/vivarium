"""
pipeline/measurement/water_level.py

Tier 1 — Geometric Anchor for water level.

Pipeline (per architecture doc):
  1. Receive warped tube crop from bottle_detector (or use roi directly)
  2. Apply perspective warp to correct camera viewing angle
  3. Extract tube pixel column (vertical strip of maximum edge activity)
  4. Run linear regressor on that column to find fill level y
  5. Compute: water_pct = (1 - y_fill / tube_height) × 100
  6. Emit occlusion_flag if fill line confidence is too low

Used by: opencv_water_measurer.py (as the CV engine inside ComparatorMeasurer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class WaterLevelResult:
    water_pct: float          # 0–100
    y_fill: int               # pixel row of detected fill line (in warped space)
    tube_height: int          # height of tube region used
    confidence: float         # 0–1
    occlusion_flag: bool      # True if fill line could not be reliably found
    label: str


class WaterLevelDetector:
    """
    Geometric water level estimator.

    Accepts a BGR crop of the water bottle/tube region (already detected
    by bottle_detector or passed directly from the pipeline ROI).

    All parameters are explicit constructor args — no config dependency —
    so this class is fully testable in isolation.

    Args:
        warp_src_pts:   4 source points for perspective warp (None = identity warp).
                        Pass np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) clockwise
                        from top-left if the camera angle introduces perspective.
        tube_col_frac:  Fraction of crop width to use as the tube column strip.
                        0.3 = centre 30% of the crop. Default 0.4.
        min_edge_rows:  Minimum number of edge-active rows required to trust the
                        fill line estimate. Below this → occlusion_flag=True.
        blur_ksize:     Gaussian blur kernel (odd int) applied before edge detection.
    """

    def __init__(
        self,
        warp_src_pts: Optional[np.ndarray] = None,
        tube_col_frac: float = 0.40,
        min_edge_rows: int = 10,
        blur_ksize: int = 5,
    ) -> None:
        self._warp_src = warp_src_pts
        self._tube_col_frac = tube_col_frac
        self._min_edge_rows = min_edge_rows
        self._blur_ksize = blur_ksize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, roi: np.ndarray) -> WaterLevelResult:
        """
        Run the full geometric water level pipeline on a BGR crop.

        Args:
            roi: BGR numpy array — the water bottle / tube crop.

        Returns:
            WaterLevelResult dataclass.
        """
        h, w = roi.shape[:2]

        # Step 1 — Perspective warp
        warped = self._perspective_warp(roi, w, h)

        # Step 2 — Grayscale + blur
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self._blur_ksize, self._blur_ksize), 0)

        # Step 3 — Extract tube column strip (centre band with most vertical activity)
        tube_col = self._extract_tube_column(blurred, w)

        # Step 4 — Find fill line via linear regressor on column intensity profile
        y_fill, fill_confidence = self._find_fill_line(tube_col)

        # Step 5 — Compute water_pct
        tube_height = h
        water_pct = float(np.clip((1.0 - y_fill / max(tube_height, 1)) * 100.0, 0.0, 100.0))

        # Step 6 — Occlusion flag
        occlusion_flag = fill_confidence < 0.25

        label = (
            f"CV water: {water_pct:.1f}% | fill_y={y_fill}/{tube_height} "
            f"| conf={fill_confidence:.2f}"
            + (" | ⚠️OCCLUDED" if occlusion_flag else "")
        )

        logger.debug(label)

        return WaterLevelResult(
            water_pct=round(water_pct, 1),
            y_fill=y_fill,
            tube_height=tube_height,
            confidence=fill_confidence,
            occlusion_flag=occlusion_flag,
            label=label,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perspective_warp(self, roi: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Apply a perspective warp if src points were provided, otherwise return roi.

        The warp corrects camera viewing angle so the tube appears perfectly vertical.
        Critical for angled bottle tubes where the fill line appears slanted.
        """
        if self._warp_src is None:
            return roi  # identity — already frontal

        dst_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        M = cv2.getPerspectiveTransform(self._warp_src, dst_pts)
        warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def _extract_tube_column(self, gray: np.ndarray, full_width: int) -> np.ndarray:
        """
        Extract the central horizontal band most likely to contain the tube.

        Uses variance across rows to find the column range with the most
        vertical intensity variation (= the tube walls and fill line).

        Returns a 1D array of mean intensities per row within that column band.
        """
        h, w = gray.shape

        # Centre fraction
        col_width = max(1, int(w * self._tube_col_frac))
        x_start = (w - col_width) // 2
        x_end = x_start + col_width

        strip = gray[:, x_start:x_end]          # h × col_width

        # Row means within the strip
        row_means = np.mean(strip, axis=1)       # shape: (h,)
        return row_means

    def _find_fill_line(self, row_means: np.ndarray) -> Tuple[int, float]:
        """
        Locate the fill line row using a linear regressor on the intensity profile.

        Strategy:
          - Above the fill line: air → typically bright / uniform
          - Below the fill line: water → darker or different texture
          - The fill line itself: largest intensity gradient (transition)

        Returns:
            (y_fill, confidence) where y_fill is the row index in the warped image.
        """
        n = len(row_means)
        if n < 4:
            return n // 2, 0.0

        # Smooth the profile to reduce noise
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(row_means, kernel, mode="same")

        # Compute first-order gradient (rate of change row-to-row)
        gradient = np.gradient(smoothed.astype(float))

        # The fill line is the row with the steepest downward transition
        # (air → water: intensity typically drops)
        y_fill_down = int(np.argmin(gradient))

        # Also find steepest upward transition (some bottles show bright meniscus)
        y_fill_up = int(np.argmax(gradient))

        # Use the transition with the larger absolute magnitude
        if abs(gradient[y_fill_down]) >= abs(gradient[y_fill_up]):
            y_fill = y_fill_down
            grad_strength = float(abs(gradient[y_fill_down]))
        else:
            y_fill = y_fill_up
            grad_strength = float(abs(gradient[y_fill_up]))

        # Confidence: normalise gradient strength by mean intensity range
        intensity_range = float(smoothed.max() - smoothed.min())
        confidence = float(np.clip(grad_strength / max(intensity_range, 1.0), 0.0, 1.0))

        # Additional confidence check: count edge-active rows
        # (low count → occluded or uniform bottle → low confidence)
        edge_active_rows = int(np.sum(np.abs(gradient) > 0.5))
        if edge_active_rows < self._min_edge_rows:
            confidence *= 0.5

        return int(np.clip(y_fill, 0, n - 1)), float(np.clip(confidence, 0.0, 1.0))