from __future__ import annotations
from dotmap import DotMap
from pipeline.annotator.base import BaseAnnotator

_REGISTRY: dict[str, str] = {
    "opencv": "pipeline.annotator.opencv_annotator.OpenCVAnnotator",
}


class AnnotatorFactory:
    @staticmethod
    def create(config: DotMap) -> BaseAnnotator:
        engine = config.annotator.engine.lower().strip()
        if engine not in _REGISTRY:
            raise ValueError(
                f"Unknown annotator engine '{engine}'. "
                f"Supported: {list(_REGISTRY.keys())}"
            )
        module_path, class_name = _REGISTRY[engine].rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(config)