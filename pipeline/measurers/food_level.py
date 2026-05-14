"""
pipeline/measurement/food_level.py

Tier 1 — Geometric Anchor for food level.

Pipeline (per architecture doc):
  1. Detect hopper bbox (roi passed in directly from orchestrator)
  2. Apply Canny edge detection with adaptive thresholds
  3. Run Hough transform → rough_surface_y (dominant horizontal line)
  4. Run reference-line regression → refined_surface_y
  5. Fuse rough + refined via confidence-weighted average (surface_fusion)
  6. Compute: food_pct = (bot_y - final_surface_y) / hopper_height × 100

Used by: opencv_food_measurer.py (as the CV engine inside ComparatorMeasurer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class FoodLevelResult:
    food_pct: float           # 0–100
    final_surface_y: int      # fused surface row in pixel space
    rough_surface_y: int      # from Hough transform
    refined_surface_y: int    # from reference-line regression
    rough_conf: float         # 0–1
    refined_conf: float       # 0–1
    confidence: float         # fused confidence
    label: str


class FoodLevelDetector:
    """
    Geometric food level estimator.

    Accepts a BGR crop of the food hopper region.
    Runs Canny + Hough + reference-line regression, fuses both estimates.

    Args:
        canny_low_frac:       Fraction of median intensity for Canny low threshold.
        canny_high_frac:      Same for high threshold.
        hough_min_line_frac:  Minimum line length as fraction of image width.
        hough_max_gap:        Max pixel gap in Hough line.
        min_hough_lines:      Minimum Hough lines required to trust rough_surface_y.
        ref_lines_y:          Known cage geometry reference rows as fractions of height.
                              e.g. [0.1, 0.9] means top-10% and bottom-10% are cage walls.
        surface_search_frac:  FIX — fraction of the interior region to search for the
                              food surface. Only the TOP portion of the interior is
                              searched, preventing bowl rims and cage bars at the bottom
                              from dominating argmax. Default 0.65 (top 65% of interior).
    """

    def __init__(
        self,
        canny_low_frac: float = 0.33,
        canny_high_frac: float = 1.0,
        hough_min_line_frac: float = 0.30,
        hough_max_gap: int = 15,
        min_hough_lines: int = 1,
        ref_lines_y: Optional[List[float]] = None,
        surface_search_frac: float = 0.65,
    ) -> None:
        self._canny_low_frac = canny_low_frac
        self._canny_high_frac = canny_high_frac
        self._hough_min_line_frac = hough_min_line_frac
        self._hough_max_gap = hough_max_gap
        self._min_hough_lines = min_hough_lines
        self._ref_lines_y = ref_lines_y or [0.05, 0.95]  # top/bottom cage walls
        self._surface_search_frac = float(np.clip(surface_search_frac, 0.2, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, roi: np.ndarray) -> FoodLevelResult:
        """
        Run the full geometric food level pipeline on a BGR hopper crop.

        Args:
            roi: BGR numpy array — the food hopper crop.

        Returns:
            FoodLevelResult dataclass.
        """
        h, w = roi.shape[:2]

        # Step 1 — Canny with adaptive thresholds
        edges = self._adaptive_canny(roi)

        # Step 2 — Hough transform → rough_surface_y
        rough_y, rough_conf = self._hough_surface(edges, w, h)

        # Step 3 — Reference-line regression → refined_surface_y
        refined_y, refined_conf = self._regression_surface(edges, h)

        # Step 4 — surface_fusion: confidence-weighted average
        final_y, fused_conf = self._surface_fusion(
            rough_y, rough_conf, refined_y, refined_conf
        )

        # Step 5 — food_pct formula (architecture doc)
        bot_y = h
        hopper_height = max(bot_y, 1)
        food_pct = float(np.clip((bot_y - final_y) / hopper_height * 100.0, 0.0, 100.0))

        label = (
            f"CV food: {food_pct:.1f}% | "
            f"rough_y={rough_y}(c={rough_conf:.2f}) "
            f"refined_y={refined_y}(c={refined_conf:.2f}) "
            f"final_y={final_y} | conf={fused_conf:.2f}"
        )
        logger.debug(label)

        return FoodLevelResult(
            food_pct=round(food_pct, 1),
            final_surface_y=final_y,
            rough_surface_y=rough_y,
            refined_surface_y=refined_y,
            rough_conf=rough_conf,
            refined_conf=refined_conf,
            confidence=fused_conf,
            label=label,
        )

    # ------------------------------------------------------------------
    # Step 1 — Adaptive Canny
    # ------------------------------------------------------------------

    def _adaptive_canny(self, roi: np.ndarray) -> np.ndarray:
        """
        Canny edge detection with thresholds adapted to image median intensity.

        Using median-based thresholds makes the detector robust to varying
        hopper illumination (bright food vs. dark food, shadows, etc.).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(blurred))
        low = int(max(0, self._canny_low_frac * median))
        high = int(max(low + 1, self._canny_high_frac * median))

        edges = cv2.Canny(blurred, low, high)
        return edges

    # ------------------------------------------------------------------
    # Step 2 — Hough → rough_surface_y
    # ------------------------------------------------------------------

    def _hough_surface(
        self, edges: np.ndarray, w: int, h: int
    ) -> Tuple[int, float]:
        """
        Detect dominant horizontal surface line using Hough transform.

        Returns the y-coordinate of the most prominent horizontal line
        (= top of food pile), plus a confidence value.
        """
        min_length = max(1, int(w * self._hough_min_line_frac))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=min_length,
            maxLineGap=self._hough_max_gap,
        )

        if lines is None:
            return h // 2, 0.15

        # Filter to horizontal lines only (vertical diff < 8px)
        h_lines = [
            line[0] for line in lines
            if abs(int(line[0][1]) - int(line[0][3])) < 8
        ]

        if len(h_lines) < self._min_hough_lines:
            return h // 2, 0.15

        # FIX: restrict Hough candidate lines to the top surface_search_frac
        # of the image so the bowl rim (near bottom of ROI) cannot be picked
        # as the food surface. The surface of a food pile is always above the
        # bowl rim when the ROI is correctly cropped to the hopper.
        search_limit_y = int(h * self._surface_search_frac)
        h_lines_filtered = [l for l in h_lines if int((l[1] + l[3]) / 2) < search_limit_y]

        # Fall back to all lines if the filter removes everything
        if not h_lines_filtered:
            logger.debug(
                "food_level: no Hough lines in top %.0f%% — using all horizontal lines",
                self._surface_search_frac * 100,
            )
            h_lines_filtered = h_lines

        y_vals = [int((l[1] + l[3]) / 2) for l in h_lines_filtered]

        # Food surface = topmost dense cluster of horizontal lines
        y_vals_sorted = sorted(y_vals)
        top_cluster = y_vals_sorted[: max(1, len(y_vals_sorted) // 3)]
        rough_y = int(np.median(top_cluster))

        conf = float(min(len(h_lines_filtered) / 8.0, 1.0))

        return int(np.clip(rough_y, 0, h - 1)), conf

    # ------------------------------------------------------------------
    # Step 3 — Reference-line regression → refined_surface_y
    # ------------------------------------------------------------------

    def _regression_surface(
        self, edges: np.ndarray, h: int
    ) -> Tuple[int, float]:
        """
        Reference-line regression against known cage geometry.

        FIX: Only searches the top surface_search_frac of the interior region
        for the peak edge density row. This prevents the bowl rim — which sits
        near the bottom of the hopper ROI and produces a strong horizontal edge
        band — from dominating argmax and being reported as the food surface.

        The food surface is always above the bowl rim, so restricting the search
        window to the upper portion of the interior is geometrically correct and
        eliminates the most common false-positive.
        """
        row_density = np.sum(edges, axis=1).astype(float)  # shape: (h,)

        # Exclude known cage wall regions
        top_wall = int(self._ref_lines_y[0] * h)
        bot_wall = int(self._ref_lines_y[-1] * h)

        interior = row_density[top_wall:bot_wall]
        if len(interior) < 2:
            return h // 2, 0.1

        # FIX: restrict peak search to top surface_search_frac of the interior.
        # Bowl rim creates a strong edge band at the bottom of the interior —
        # searching only the upper portion prevents it from hijacking argmax.
        search_end = max(2, int(len(interior) * self._surface_search_frac))
        search_region = interior[:search_end]

        peak_idx = int(np.argmax(search_region))
        refined_y = top_wall + peak_idx

        # Confidence: peak density relative to mean of the search region
        mean_density = float(np.mean(search_region)) + 1e-6
        peak_density = float(search_region[peak_idx])
        conf = float(np.clip((peak_density / mean_density - 1.0) / 3.0, 0.0, 1.0))

        return int(np.clip(refined_y, 0, h - 1)), conf

    # ------------------------------------------------------------------
    # Step 4 — surface_fusion: confidence-weighted average
    # ------------------------------------------------------------------

    def _surface_fusion(
        self,
        rough_y: int, rough_conf: float,
        refined_y: int, refined_conf: float,
    ) -> Tuple[int, float]:
        """
        Combines rough (Hough) + refined (regression) using confidence-weighted
        average. Agreement between estimates boosts confidence; disagreement
        penalises it.
        """
        total = rough_conf + refined_conf
        if total < 1e-6:
            return (rough_y + refined_y) // 2, 0.1

        fused_y = int(
            (rough_y * rough_conf + refined_y * refined_conf) / total
        )

        gap = abs(rough_y - refined_y)
        if gap < 10:
            fused_conf = min((rough_conf + refined_conf) / 2.0 * 1.2, 1.0)
        elif gap < 25:
            fused_conf = (rough_conf + refined_conf) / 2.0
        else:
            fused_conf = (rough_conf + refined_conf) / 2.0 * 0.6

        return fused_y, float(np.clip(fused_conf, 0.0, 1.0))