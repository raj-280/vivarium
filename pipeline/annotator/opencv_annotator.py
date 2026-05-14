"""
pipeline/annotator/opencv_annotator.py

Draws bounding boxes and measurement labels onto the preprocessed image.
Handles water, food, mouse, and bedding targets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger

from core.result import BoundingBox, MeasurementResult
from pipeline.annotator.base import BaseAnnotator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# BGR colours per target
_COLORS = {
    "water":   (255, 100,   0),   # blue-orange
    "food":    (  0, 200,  50),   # green
    "mouse":   (  0, 220, 255),   # yellow-cyan
    "bedding": (180,  60, 255),   # purple
}
_DEFAULT_COLOR = (200, 200, 200)

# Bedding condition → BGR colour for the label background
_BEDDING_COLORS = {
    "PERFECT": ( 80, 200,  80),
    "OK":      (  0, 180, 255),
    "BAD":     (  0, 100, 255),
    "WORST":   (  0,   0, 255),
    "NOT_DETECTED": (160, 160, 160),
}


class OpenCVAnnotator(BaseAnnotator):

    def draw(
        self,
        image: np.ndarray,
        gated: Dict[str, Optional[BoundingBox]],
        measurements: Dict[str, MeasurementResult],
        result_id: str,
        filename: str,
    ) -> str:
        out = image.copy()
        h, w = out.shape[:2]

        drawn_count = 0
        for target, bbox in gated.items():
            if bbox is None:
                continue

            x1 = int(bbox.x1 * w)
            y1 = int(bbox.y1 * h)
            x2 = int(bbox.x2 * w)
            y2 = int(bbox.y2 * h)

            meas = measurements.get(target)

            # ── Colour selection ──────────────────────────────────────
            if target == "bedding" and meas is not None and meas.bedding_condition:
                color = _BEDDING_COLORS.get(meas.bedding_condition, _DEFAULT_COLOR)
            else:
                color = _COLORS.get(target, _DEFAULT_COLOR)

            # ── Bounding box ──────────────────────────────────────────
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # ── Label text ────────────────────────────────────────────
            if target == "mouse":
                label = f"mouse: {'present' if meas and meas.present else 'absent'}"
            elif target == "bedding":
                if meas and meas.bedding_condition:
                    label = f"bedding: {meas.bedding_condition}"
                else:
                    label = "bedding: unknown"
            elif meas and meas.label and meas.label not in ("UNDETERMINED", "NOT_DETECTED", "UNKNOWN"):
                # Use the human-readable label from measurer e.g. "food OK (cls7)"
                # Extract just the name part between target and (cls...)
                name = meas.label.split(" ")[1] if " " in meas.label else meas.label
                label = f"{target}: {name} ({meas.level:.0f}%)"
            elif meas:
                label = f"{target}: uncertain"
            else:
                label = target

            # Append confidence to label
            label = f"{label} ({bbox.confidence:.2f})"

            # ── Label placement ───────────────────────────────────────
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            pad = 4
            lx = min(x1, w - tw - pad * 2)
            lx = max(lx, 0)

            if y1 - th - 8 >= 0:
                bg_y1, bg_y2, text_y = y1 - th - 8, y1, y1 - pad
            else:
                bg_y1, bg_y2, text_y = y2, y2 + th + 8, y2 + th + pad

            cv2.rectangle(out, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), color, -1)
            cv2.putText(
                out, label, (lx + pad, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            drawn_count += 1

        logger.debug(f"Annotator drew {drawn_count} boxes on {w}x{h} image")

        raw_preview_dir = str(self.config.annotator.preview_dir)
        if os.path.isabs(raw_preview_dir):
            preview_dir = Path(raw_preview_dir)
        else:
            preview_dir = _PROJECT_ROOT / raw_preview_dir
        preview_dir = preview_dir.resolve()

        os.makedirs(preview_dir, exist_ok=True)

        base = os.path.splitext(filename)[0]
        out_filename = f"{result_id}_{base}.jpg"
        out_path = str(preview_dir / out_filename)

        success = cv2.imwrite(out_path, out)
        if not success:
            logger.error(f"cv2.imwrite FAILED for path '{out_path}'")
            raise IOError(f"cv2.imwrite failed to save annotated image to '{out_path}'")

        logger.info(f"Annotator saved image → {out_path}")
        return out_path
