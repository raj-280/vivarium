from __future__ import annotations
import importlib
from dotmap import DotMap
from loguru import logger
from pipeline.annotator.base import BaseAnnotator

_REGISTRY: dict[str, str] = {
    "opencv": "pipeline.annotator.opencv_annotator.OpenCVAnnotator",
}


class AnnotatorFactory:
    @staticmethod
    def create(config: DotMap) -> BaseAnnotator:
        engine = config.annotator.engine.lower().strip()
        if engine not in _REGISTRY:
            logger.error(f"Unknown annotator engine '{engine}'. Supported: {list(_REGISTRY.keys())}")
            raise ValueError(
                f"Unknown annotator engine '{engine}'. "
                f"Supported: {list(_REGISTRY.keys())}"
            )
        module_path, class_name = _REGISTRY[engine].rsplit(".", 1)
        logger.debug(f"Creating annotator | engine={engine}")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(config)