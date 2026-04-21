"""
pipeline/storage/factory.py

Returns the correct BaseStorage implementation based on storage.engine config key.
Also returns the correct BaseImageStore based on storage.image_store config key.
"""

from __future__ import annotations

import importlib

from dotmap import DotMap

from pipeline.storage.base import BaseImageStore, BaseStorage


class ConfigurationError(Exception):
    """Raised when an unsupported engine value is found in config."""


_DB_REGISTRY: dict[str, str] = {
    "postgres": "pipeline.storage.postgres.PostgresStorage",
}

_IMAGE_STORE_REGISTRY: dict[str, str] = {
    "local": "pipeline.storage.image_store.local.LocalImageStore",
    "s3": "pipeline.storage.image_store.s3.S3ImageStore",
    "gcs": "pipeline.storage.image_store.gcs.GCSImageStore",
}


class StorageFactory:
    """Factory for database storage and image store implementations."""

    @staticmethod
    def create_db(config: DotMap) -> BaseStorage:
        """
        Return the database storage implementation from config.storage.engine.

        Raises:
            ConfigurationError: If engine is unknown.
        """
        engine: str = config.storage.engine.lower().strip()

        if engine not in _DB_REGISTRY:
            supported = ", ".join(_DB_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown storage engine '{engine}'. "
                f"Supported: {supported}. "
                f"Check config.yaml → storage.engine"
            )

        module_path, class_name = _DB_REGISTRY[engine].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config)

    @staticmethod
    def create_image_store(config: DotMap) -> BaseImageStore:
        """
        Return the image store implementation from config.storage.image_store.

        Raises:
            ConfigurationError: If image_store value is unknown.
        """
        store: str = config.storage.image_store.lower().strip()

        if store not in _IMAGE_STORE_REGISTRY:
            supported = ", ".join(_IMAGE_STORE_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown image store '{store}'. "
                f"Supported: {supported}. "
                f"Check config.yaml → storage.image_store"
            )

        module_path, class_name = _IMAGE_STORE_REGISTRY[store].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config)
