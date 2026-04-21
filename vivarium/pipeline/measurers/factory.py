"""
pipeline/measurers/factory.py

Returns the correct BaseMeasurer subclass per target based on
{target}.engine config key.

Callers must never import measurer implementations directly.
"""

from __future__ import annotations

import importlib

from dotmap import DotMap

from pipeline.measurers.base import BaseMeasurer


class ConfigurationError(Exception):
    """Raised when an unsupported engine value is found in config."""


_REGISTRY: dict[str, str] = {
    "clip": "pipeline.measurers.clip_measurer.CLIPMeasurer",
    "opencv": "pipeline.measurers.opencv_measurer.OpenCVMeasurer",
    "classifier": "pipeline.measurers.classifier.ClassifierMeasurer",
    "yolo": "pipeline.measurers.clip_measurer.CLIPMeasurer",  # mouse yolo engine falls back to CLIP presence check
}


class MeasurerFactory:
    """Factory that instantiates the correct measurer per target from config."""

    @staticmethod
    def create(config: DotMap, target: str) -> BaseMeasurer:
        """
        Instantiate and return the measurer for the given target.

        Args:
            config: Full DotMap configuration object.
            target: Target name — "water", "food", or "mouse".

        Returns:
            An initialised (but not yet loaded) BaseMeasurer subclass instance.

        Raises:
            ConfigurationError: If the engine value for this target is unknown.
        """
        target_cfg = getattr(config, target, None)
        if target_cfg is None:
            raise ConfigurationError(
                f"No configuration section found for target '{target}'. "
                f"Add a '{target}:' block to config.yaml."
            )

        engine: str = target_cfg.engine.lower().strip()

        if engine not in _REGISTRY:
            supported = ", ".join(_REGISTRY.keys())
            raise ConfigurationError(
                f"Unknown measurer engine '{engine}' for target '{target}'. "
                f"Supported engines: {supported}. "
                f"Check config.yaml → {target}.engine"
            )

        module_path, class_name = _REGISTRY[engine].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config, target)
