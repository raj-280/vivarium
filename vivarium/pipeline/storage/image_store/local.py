"""
pipeline/storage/image_store/local.py

Local filesystem image store implementation.
Saves images to config.storage.local_image_path directory.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from dotmap import DotMap
from loguru import logger

from pipeline.storage.base import BaseImageStore


class LocalImageStore(BaseImageStore):
    """Saves images to the local filesystem."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._base_path = Path(config.storage.local_image_path)

    async def save(self, image_bytes: bytes, filename: str) -> str:
        """
        Write image bytes to disk. Generates a UUID-prefixed filename if needed.

        Returns:
            Absolute path string where the image was saved.
        """
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Prepend UUID to prevent collisions
        safe_name = f"{uuid.uuid4().hex}_{filename}"
        dest = self._base_path / safe_name

        dest.write_bytes(image_bytes)
        logger.debug(f"Image saved locally → {dest}")
        return str(dest)

    async def get_url(self, stored_path: str) -> str:
        """Return the local filesystem path as-is."""
        return stored_path
