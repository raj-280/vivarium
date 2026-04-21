"""
pipeline/measurers/factory.py

Returns the correct BaseMeasurer subclass per target based on
{target}.engine config key.

FIX: Removed the silent "yolo" → CLIPMeasurer fallback.
     "yolo" was never a valid measurer engine — it's a detector engine.
     Setting mouse.engine = "yolo" in config previously loaded CLIP silently,
     giving wrong results with no warning. Now it raises ConfigurationError clearly.
     Use mouse.engine = "clip" for CLIP-based mouse presence detection.
"""

from __future__ import annotations

import importlib

from dotmap import DotMap

from pipeline.measurers.base import BaseMeasurer


class ConfigurationError(Exception):
    """Raised when an unsupported engine value is found in config."""


# Registry maps config engine name → fully qualified class path.
# To add a new measurer: add one entry here + create the implementation file.
# "yolo" is intentionally NOT in this registry — it is a detector engine, not a measurer.
_REGISTRY: dict[str, str] = {
    "clip":       "pipeline.measurers.clip_measurer.CLIPMeasurer",
    "opencv":     "pipeline.measurers.opencv_measurer.OpenCVMeasurer",
    "classifier": "pipeline.measurers.classifier.ClassifierMeasurer",
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
            ConfigurationError: If the engine value for this target is unknown,
                                 or if no config section exists for this target.
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
                f"Check config.yaml → {target}.engine\n"
                f"Note: 'yolo' is a detector engine, not a measurer engine. "
                f"For mouse presence detection use 'clip'."
            )

        module_path, class_name = _REGISTRY[engine].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(config, target)