"""
pipeline/storage/image_store/s3.py

AWS S3 image store implementation.
Reads bucket, region, and prefix from config.storage.s3.
Credentials are loaded from environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
"""

from __future__ import annotations

import uuid

from dotmap import DotMap
from loguru import logger

from pipeline.storage.base import BaseImageStore


class S3ImageStore(BaseImageStore):
    """Saves images to AWS S3."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        s3_cfg = config.storage.s3
        self._bucket: str = s3_cfg.bucket
        self._region: str = s3_cfg.region
        self._prefix: str = s3_cfg.prefix

    async def save(self, image_bytes: bytes, filename: str) -> str:
        """
        Upload image bytes to S3.

        Returns:
            S3 key path (s3://bucket/prefix/filename).
        """
        try:
            import aioboto3  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "aioboto3 not installed. Run: pip install aioboto3"
            ) from exc

        key = f"{self._prefix.rstrip('/')}/{uuid.uuid4().hex}_{filename}"

        async with aioboto3.Session().client(
            "s3", region_name=self._region
        ) as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=image_bytes,
                ContentType="image/jpeg",
            )

        s3_path = f"s3://{self._bucket}/{key}"
        logger.debug(f"Image uploaded to S3 → {s3_path}")
        return s3_path

    async def get_url(self, stored_path: str) -> str:
        """Return HTTPS URL for the S3 object."""
        # stored_path is like s3://bucket/key
        key = stored_path.removeprefix(f"s3://{self._bucket}/")
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"
