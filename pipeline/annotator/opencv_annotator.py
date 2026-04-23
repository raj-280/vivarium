from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox, MeasurementResult
from pipeline.annotator.base import BaseAnnotator

# BGR colors per target
_COLORS = {
    "water": (255, 100,   0),
    "food":  (  0, 200,  50),
    "mouse": (  0, 220, 255),
}
_DEFAULT_COLOR = (200, 200, 200)


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

        for target, bbox in gated.items():
            if bbox is None:
                continue

            x1 = int(bbox.x1 * w)
            y1 = int(bbox.y1 * h)
            x2 = int(bbox.x2 * w)
            y2 = int(bbox.y2 * h)
            color = _COLORS.get(target, _DEFAULT_COLOR)

            # Draw bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Build label text
            meas = measurements.get(target)
            if target == "mouse":
                label = f"mouse: {'present' if meas and meas.present else 'absent'}"
            elif meas:
                label = f"{target}: {meas.level:.0f}%"
            else:
                label = target

            # Draw label background + text
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                out,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color, -1
            )
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # Save to outputs/annotated/<result_id>_<filename>
        preview_dir = str(self.config.annotator.preview_dir)
        os.makedirs(preview_dir, exist_ok=True)

        base = os.path.splitext(filename)[0]
        out_filename = f"{result_id}_{base}.jpg"
        out_path = os.path.join(preview_dir, out_filename)

        cv2.imwrite(out_path, out)
        logger.info(f"Annotator saved image → {out_path}")

        return out_path