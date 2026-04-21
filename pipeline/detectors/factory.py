"""
pipeline/detectors/factory.py

Returns the correct BaseDetector subclass based on detector.engine config key.
Callers must never import detector implementations directly.
"""

from __future__ import annotations

from dotmap import DotMap

from pipeline.detectors.base import BaseDetector


class ConfigurationError(Exception):
    """Raised when an unsupported engine value is found in config."""


_REGISTRY: dict[str, str] = {
    "yolov8world": "pipeline.detectors.yolov8world.YOLOv8WorldDetector",
    "groundingdino": "pipeline.detectors.groundingdino.GroundingDINODetector",
}


class DetectorFactory:
    """Factory that instantiates the correct detector from config."""

    @staticmethod
    def create(config: DotMap) -> BaseDetector:
        """
        Instantiate and return the detector specified by config.detector.engine.

        Args:
            config: Full DotMap configuration object.

        Returns:
            An initialised (but not yet loaded) BaseDetector subclass instance.

        Raises:
            ConfigurationError: If detector.engine value is unknown.
        """
        engine: str = config.detector.engine.lower().strip()

        if engine not in _REGISTRY:
            supported = ", ".join(_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown detector engine '{engine}'. "
                f"Supported engines: {supported}. "
                f"Check config.yaml → detector.engine"
            )

        # Lazy import to avoid loading heavyweight models at import time
        module_path, class_name = _REGISTRY[engine].rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config)
