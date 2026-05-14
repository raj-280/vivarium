"""
pipeline/measurers/factory.py

MeasurerFactory — creates the right BaseMeasurer subclass for a given engine key.
"""

from __future__ import annotations

import importlib

from dotmap import DotMap

from pipeline.measurers.base import BaseMeasurer


class ConfigurationError(Exception):
    pass


class MeasurerFactory:

    _REGISTRY: dict[str, tuple[str, str]] = {
        "opencv_water":       ("pipeline.measurers.opencv_water_measurer",         "OpenCVWaterMeasurer"),
        "opencv_food":        ("pipeline.measurers.opencv_food_measurer",           "OpenCVFoodMeasurer"),
        "unet_water":         ("pipeline.measurers.unet_measurer",                 "UNetWaterMeasurer"),
        "unet_food":          ("pipeline.measurers.unet_measurer",                 "UNetFoodMeasurer"),
        "classifier":         ("pipeline.measurers.classifier",                    "ClassifierMeasurer"),
        "fcn_psp":            ("pipeline.measurers.pspnet_measurer",               "PSPNetWaterMeasurer"),
        "detection_presence": ("pipeline.measurers.detection_presence_measurer",   "DetectionPresenceMeasurer"),
        # YOLOX: decodes level/condition directly from class ID — no second model
        "yolox_class":        ("pipeline.measurers.yolox_measurer",                "YOLOXMeasurer"),
    }

    @classmethod
    def create(cls, config: DotMap, target: str) -> BaseMeasurer:
        target_cfg = getattr(config.measurers, target, None)
        if target_cfg is None:
            raise ConfigurationError(
                f"No config section found for measurers.{target}. "
                f"Add a '{target}:' block under 'measurers:' in config.yaml."
            )

        engine = getattr(target_cfg, "engine", None)
        if engine is None:
            raise ConfigurationError(
                f"No engine set for measurers.{target}. "
                f"Set measurers.{target}.engine in config.yaml."
            )

        if engine not in cls._REGISTRY:
            registered = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ConfigurationError(
                f"Unknown measurer engine '{engine}' for target '{target}'. "
                f"Registered engines: {registered}"
            )

        module_path, class_name = cls._REGISTRY[engine]
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name)
        return klass(config=config, target=target)

    @classmethod
    def register(cls, key: str, module_path: str, class_name: str) -> None:
        cls._REGISTRY[key] = (module_path, class_name)
