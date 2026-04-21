"""
pipeline/measurers/opencv_measurer.py

OpenCV edge-detection measurer implementation.
Estimates fill level by measuring the proportion of detected edges
below the vertical midpoint of the ROI. All parameters from config.
"""

from __future__ import annotations

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


class OpenCVMeasurer(BaseMeasurer):
    """Level estimator using OpenCV edge detection (Canny or Sobel)."""

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)

    def load(self) -> None:
        """No model to load for OpenCV — verify cv2 is importable."""
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "opencv-python is not installed. Run: pip install opencv-python"
            ) from exc
        logger.info(f"OpenCVMeasurer ready for target '{self.target}'")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Estimate level by edge detection + optional horizontal line detection.

        Strategy:
          1. Convert ROI to grayscale.
          2. Apply Canny (or Sobel) edge detection.
          3. (Optional) Detect horizontal lines using Hough transform for water meniscus.
          4. Compute edge density in the bottom half vs. full image.
          5. Use the ratio as a proxy for fill level.

        Returns:
            MeasurementResult with level (0–100), confidence, label.
        """
        import cv2  # type: ignore

        target_cfg = getattr(self.config, self.target)
        opencv_cfg = target_cfg.opencv

        edge_method: str = opencv_cfg.edge_method.lower()
        enable_hlines: bool = getattr(opencv_cfg, 'enable_horizontal_line_detection', False)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if edge_method == "canny":
            t1 = int(opencv_cfg.canny_threshold1)
            t2 = int(opencv_cfg.canny_threshold2)
            edges = cv2.Canny(gray, t1, t2)
        elif edge_method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.uint8(np.sqrt(sobelx**2 + sobely**2))
            _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(
                f"Unsupported edge_method '{edge_method}' for target '{self.target}'. "
                "Use 'canny' or 'sobel' in config."
            )

        h = edges.shape[0]
        w = edges.shape[1]
        total_pixels = edges.size
        edge_pixels = np.count_nonzero(edges)

        # **OPTIMIZATION: Horizontal line detection for water meniscus**
        water_level_y = None
        if enable_hlines and edge_pixels > 50:
            try:
                lines = cv2.HoughLinesP(
                    edges,
                    rho=1,
                    theta=np.pi / 180,
                    threshold=30,
                    minLineLength=w // 2,
                    maxLineGap=10,
                )
                if lines is not None and len(lines) > 0:
                    # Find the most prominent horizontal line (water surface)
                    horizontal_lines = [
                        line[0] for line in lines
                        if abs(line[0][1] - line[0][3]) < 5  # vertical diff < 5 pixels
                    ]
                    if horizontal_lines:
                        # Use the lowest (most prominent) horizontal line as water level
                        water_level_y = max([line[1] for line in horizontal_lines])
                        logger.debug(
                            f"OpenCV [{target_cfg}] Detected water meniscus at y={water_level_y}"
                        )
            except Exception as e:
                logger.debug(f"Hough line detection failed: {e}")

        # Bottom-half edge density proxy: more edges below mid = more full
        if water_level_y is not None:
            # Use detected water level instead of midpoint
            bottom_half = edges[water_level_y:, :]
        else:
            bottom_half = edges[h // 2:, :]

        bottom_edge_pixels = np.count_nonzero(bottom_half)

        # Normalise: ratio of bottom edges to total edges (0 → 1)
        edge_ratio = bottom_edge_pixels / max(edge_pixels, 1)
        level = round(edge_ratio * 100.0, 1)

        # Confidence: based on how many edges were found relative to ROI size
        edge_density = edge_pixels / max(total_pixels, 1)
        confidence = float(min(edge_density * 10.0, 1.0))  # heuristic scaling

        label = f"Estimated level via {edge_method.upper()}: {level:.0f}%"

        logger.debug(
            f"OpenCV [{target_cfg}] → level={level:.1f} conf={confidence:.3f} "
            f"edge_density={edge_density:.4f}"
        )

        return MeasurementResult(
            level=level,
            confidence=confidence,
            label=label,
            present=None,
        )
