"""
pipeline/preprocessor/image_validator.py

Validates incoming image bytes:
  - Format check (against allowed_formats in config)
  - Size check (against input.max_image_size_mb in config)
  - Blur detection (Laplacian variance vs preprocessor.blur_threshold)
"""

from __future__ import annotations

import imghdr
import io
from typing import Tuple

import cv2
import numpy as np
from dotmap import DotMap
from loguru import logger


class ImageValidationError(Exception):
    """Raised when an image fails validation — contains the rejection reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ImageValidator:
    """Validates image bytes before they enter the preprocessing pipeline."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    def validate(self, image_bytes: bytes) -> np.ndarray:
        """
        Run all validation checks on raw image bytes.

        Args:
            image_bytes: Raw bytes from the upload.

        Returns:
            Decoded BGR numpy array ready for preprocessing.

        Raises:
            ImageValidationError: On any failed check.
        """
        self._check_size(image_bytes)
        self._check_format(image_bytes)
        array = self._decode(image_bytes)
        self._check_blur(array)
        return array

    def _check_size(self, image_bytes: bytes) -> None:
        max_mb: float = float(self.config.input.max_image_size_mb)
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > max_mb:
            reason = f"Image size {size_mb:.2f} MB exceeds limit of {max_mb} MB"
            logger.warning(reason)
            raise ImageValidationError(reason)

    def _check_format(self, image_bytes: bytes) -> None:
        allowed: list[str] = list(self.config.input.allowed_formats)
        detected = imghdr.what(None, h=image_bytes)
        # imghdr returns "jpeg" not "jpg"
        if detected == "jpeg":
            detected = "jpeg"
        # Allow both jpg and jpeg
        allowed_normalised = {f.lower().replace("jpg", "jpeg") for f in allowed}
        if detected not in allowed_normalised:
            reason = (
                f"Image format '{detected}' not in allowed formats: "
                f"{list(self.config.input.allowed_formats)}"
            )
            logger.warning(reason)
            raise ImageValidationError(reason)

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            reason = "Failed to decode image — file may be corrupt"
            logger.warning(reason)
            raise ImageValidationError(reason)
        return img

    def _check_blur(self, image: np.ndarray) -> None:
        threshold: float = float(self.config.preprocessor.blur_threshold)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        logger.debug(f"Blur check: laplacian_var={laplacian_var:.2f} threshold={threshold}")
        if laplacian_var < threshold:
            reason = (
                f"Image is too blurry (Laplacian variance {laplacian_var:.1f} "
                f"< threshold {threshold})"
            )
            logger.warning(reason)
            raise ImageValidationError(reason)
