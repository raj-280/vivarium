"""
pipeline/storage/image_store/gcs.py

Google Cloud Storage image store implementation.
Reads bucket and prefix from config.storage.gcs.
Credentials are loaded from GOOGLE_APPLICATION_CREDENTIALS env var.
"""

from __future__ import annotations

import uuid

from dotmap import DotMap
from loguru import logger

from pipeline.storage.base import BaseImageStore


class GCSImageStore(BaseImageStore):
    """Saves images to Google Cloud Storage."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        gcs_cfg = config.storage.gcs
        self._bucket: str = gcs_cfg.bucket
        self._prefix: str = gcs_cfg.prefix

    async def save(self, image_bytes: bytes, filename: str) -> str:
        """
        Upload image bytes to GCS.

        Returns:
            GCS path (gs://bucket/prefix/filename).
        """
        try:
            from google.cloud import storage as gcs_storage  # type: ignore
            import asyncio
        except ImportError as exc:
            raise ImportError(
                "google-cloud-storage not installed. "
                "Run: pip install google-cloud-storage"
            ) from exc

        key = f"{self._prefix.rstrip('/')}/{uuid.uuid4().hex}_{filename}"

        loop = asyncio.get_event_loop()

        def _upload() -> None:
            client = gcs_storage.Client()
            bucket = client.bucket(self._bucket)
            blob = bucket.blob(key)
            blob.upload_from_string(image_bytes, content_type="image/jpeg")

        await loop.run_in_executor(None, _upload)

        gcs_path = f"gs://{self._bucket}/{key}"
        logger.debug(f"Image uploaded to GCS → {gcs_path}")
        return gcs_path

    async def get_url(self, stored_path: str) -> str:
        """Return HTTPS URL for the GCS object."""
        key = stored_path.removeprefix(f"gs://{self._bucket}/")
        return f"https://storage.googleapis.com/{self._bucket}/{key}"
