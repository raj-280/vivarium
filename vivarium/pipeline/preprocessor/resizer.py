"""
pipeline/preprocessor/resizer.py

Resizes and optionally normalises images for detector input.
All parameters (resize_to, normalize) read from config.preprocessor.
"""

from __future__ import annotations

import numpy as np
import cv2
from dotmap import DotMap
from loguru import logger


class ImageResizer:
    """Resizes preprocessed images to the configured target dimensions."""

    def __init__(self, config: DotMap) -> None:
        self.config = config
        resize_cfg = config.preprocessor.resize_to
        self._target_w: int = int(resize_cfg[0])
        self._target_h: int = int(resize_cfg[1])
        self._normalize: bool = bool(config.preprocessor.normalize)

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to configured dimensions.

        Args:
            image: BGR numpy array.

        Returns:
            Resized BGR numpy array (float32 if normalized, else uint8).
        """
        resized = cv2.resize(
            image,
            (self._target_w, self._target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        logger.debug(
            f"Image resized: {image.shape[:2]} → ({self._target_h}, {self._target_w})"
        )

        if self._normalize:
            resized = resized.astype(np.float32) / 255.0
            # Denormalize back to uint8 for downstream OpenCV ops
            resized = (resized * 255).astype(np.uint8)

        return resized
