"""
pipeline/preprocessor/resizer.py

Resizes, optionally normalises, and optionally brightness-balances images
for detector input.

Preprocessing order:
  1. Resize to target dimensions (always)
  2. Brightness balancing via CLAHE (if preprocessor.brightness_balance: true)
  3. Normalize to [0,1] then back to uint8 (if preprocessor.normalize: true)

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied per
channel in LAB colour space so hue is preserved while luminance is
equalized. This handles:
  - Dark night-cycle frames
  - Overexposed daytime frames
  - Uneven cage lighting

Config keys (all under preprocessor:):
  resize_to:           [640, 640]   target width x height
  normalize:           true         divide by 255 then back to uint8
  brightness_balance:  true         apply CLAHE (default: false)
  clahe_clip_limit:    2.0          CLAHE clip limit (default: 2.0)
  clahe_tile_grid:     8            CLAHE tile grid size NxN (default: 8)
"""

from __future__ import annotations

import numpy as np
import cv2
from dotmap import DotMap
from loguru import logger


class ImageResizer:
    """Resizes, brightness-balances, and optionally normalises images."""

    def __init__(self, config: DotMap) -> None:
        self.config = config
        resize_cfg = config.preprocessor.resize_to
        self._target_w: int = int(resize_cfg[0])
        self._target_h: int = int(resize_cfg[1])
        self._normalize: bool = bool(config.preprocessor.normalize)

        # Brightness balancing config
        pre = config.preprocessor
        self._brightness_balance: bool = bool(getattr(pre, "brightness_balance", False))
        clip_limit: float = float(getattr(pre, "clahe_clip_limit", 2.0))
        tile_grid: int = int(getattr(pre, "clahe_tile_grid", 8))

        # Build CLAHE object once — reused per frame (thread-safe read-only after init)
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid, tile_grid),
        )

        logger.info(
            f"ImageResizer ready | size=({self._target_w}x{self._target_h}) "
            f"normalize={self._normalize} "
            f"brightness_balance={self._brightness_balance}"
            + (
                f" clahe(clip={clip_limit} tile={tile_grid}x{tile_grid})"
                if self._brightness_balance
                else ""
            )
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize → brightness balance → normalize.

        Args:
            image: BGR uint8 numpy array from the validator.

        Returns:
            Processed BGR uint8 numpy array ready for detection.
        """
        # Step 1 — Resize
        resized = cv2.resize(
            image,
            (self._target_w, self._target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        logger.debug(
            f"Image resized: {image.shape[:2]} → ({self._target_h}, {self._target_w})"
        )

        # Step 2 — Brightness balancing (CLAHE on L channel in LAB space)
        if self._brightness_balance:
            resized = self._apply_clahe(resized)

        # Step 3 — Normalize (divide by 255, then back to uint8 for downstream CV)
        if self._normalize:
            resized = resized.astype(np.float32) / 255.0
            resized = (resized * 255).astype(np.uint8)

        return resized

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L (luminance) channel of the image in LAB colour space.

        Working in LAB means only brightness is equalized — hue and saturation
        are untouched, so food/water colour cues are preserved for the measurers.

        Args:
            image: BGR uint8 numpy array.

        Returns:
            Brightness-equalized BGR uint8 numpy array.
        """
        # BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Equalize only the L channel
        l_eq = self._clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        logger.debug(
            f"CLAHE applied | L mean before={l_channel.mean():.1f} "
            f"after={l_eq.mean():.1f}"
        )
        return result
