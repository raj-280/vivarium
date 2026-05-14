"""
core/s3_fetcher.py

Downloads an image from AWS S3 and returns the raw bytes.

Two ways to point at an image:
  1. Full S3 URI:   s3://my-bucket/cage_1/frame_001.jpg
  2. Bucket + key:  bucket="my-bucket", key="cage_1/frame_001.jpg"

Credentials are read from the standard AWS credential chain:
  env vars → ~/.aws/credentials → IAM role — no hardcoded secrets.

Config block (config.yaml → s3):
    s3:
      region: us-east-1        # overrides AWS_REGION env var if set
      bucket: ""               # default bucket (can be overridden per-request)
      presigned_url_expiry: 900  # seconds — used if you ever generate download URLs

Requires:
    pip install boto3
"""

from __future__ import annotations

from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotmap import DotMap
from loguru import logger


class S3FetchError(Exception):
    """Raised when an S3 image cannot be fetched."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class S3ImageFetcher:
    """
    Fetches raw image bytes from AWS S3.

    Instantiate once at startup, call .fetch() per request.
    """

    def __init__(self, config: DotMap) -> None:
        s3_cfg = getattr(config, "s3", DotMap())
        self._default_bucket: str = str(getattr(s3_cfg, "bucket", "") or "")
        region: str = str(getattr(s3_cfg, "region", "") or "")

        try:
            import boto3  # noqa: F811
            kwargs = {}
            if region:
                kwargs["region_name"] = region
            self._client = boto3.client("s3", **kwargs)
            logger.info(
                f"S3ImageFetcher initialised | region={region or 'from-env'} "
                f"| default_bucket='{self._default_bucket}'"
            )
        except ImportError as exc:
            raise ImportError(
                "boto3 is not installed. Run: pip install boto3"
            ) from exc

    def fetch(
        self,
        *,
        s3_uri: str | None = None,
        bucket: str | None = None,
        key: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Download an image from S3 and return (image_bytes, filename).

        Provide either:
          - s3_uri="s3://bucket/path/to/image.jpg"
          - bucket="my-bucket", key="path/to/image.jpg"

        Returns:
            Tuple of (raw image bytes, filename derived from the key).

        Raises:
            S3FetchError: on any S3 / network error.
            ValueError:   if neither s3_uri nor bucket+key is provided.
        """
        resolved_bucket, resolved_key = self._resolve(s3_uri, bucket, key)
        filename = resolved_key.split("/")[-1] or "s3_image.jpg"

        logger.info(f"S3 fetch | bucket={resolved_bucket} | key={resolved_key}")

        try:
            response = self._client.get_object(Bucket=resolved_bucket, Key=resolved_key)
            image_bytes: bytes = response["Body"].read()
            logger.info(
                f"S3 fetch OK | bucket={resolved_bucket} | key={resolved_key} "
                f"| bytes={len(image_bytes)}"
            )
            return image_bytes, filename

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"]["Message"]
            if code in ("NoSuchKey", "404"):
                raise S3FetchError(
                    f"Object not found: s3://{resolved_bucket}/{resolved_key}"
                ) from exc
            if code in ("AccessDenied", "403"):
                raise S3FetchError(
                    f"Access denied: s3://{resolved_bucket}/{resolved_key} — check IAM permissions"
                ) from exc
            raise S3FetchError(f"S3 ClientError [{code}]: {msg}") from exc

        except BotoCoreError as exc:
            raise S3FetchError(f"S3 connection error: {exc}") from exc

    def _resolve(
        self,
        s3_uri: str | None,
        bucket: str | None,
        key: str | None,
    ) -> tuple[str, str]:
        """Resolve bucket + key from either a URI or explicit params."""
        if s3_uri:
            parsed = urlparse(s3_uri)
            if parsed.scheme != "s3":
                raise ValueError(f"Invalid S3 URI scheme: '{s3_uri}'. Must start with s3://")
            resolved_bucket = parsed.netloc
            resolved_key = parsed.path.lstrip("/")
            if not resolved_bucket or not resolved_key:
                raise ValueError(f"Cannot parse bucket/key from URI: '{s3_uri}'")
            return resolved_bucket, resolved_key

        if key:
            resolved_bucket = bucket or self._default_bucket
            if not resolved_bucket:
                raise ValueError(
                    "No bucket specified. Provide 'bucket' param or set s3.bucket in config.yaml"
                )
            return resolved_bucket, key

        raise ValueError("Provide either 's3_uri' or 'key' (+ optional 'bucket')")
