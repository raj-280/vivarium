"""
core/pipeline_logger.py

Structured pipeline event logger.
Writes one JSON-Lines (.jsonl) record per pipeline run.

YOLOX update: Added bedding fields to to_log_record().
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox, PipelineResult


class PipelineRunContext:
    def __init__(
        self,
        *,
        cage_id: str,
        filename: str,
        image_size_bytes: int,
        image_received_at: datetime,
    ) -> None:
        self.cage_id = cage_id
        self.filename = filename
        self.image_size_bytes = image_size_bytes
        self.image_received_at: datetime = image_received_at
        self.image_processed_at: Optional[datetime] = None
        self.total_processing_ms: Optional[float] = None
        self.result: Optional[PipelineResult] = None
        self.mouse_previous_bbox: Optional[BoundingBox] = None
        self.mouse_consecutive_count: Optional[int] = None
        self.duplicate_frame: bool = False
        self.webhook_fired: bool = False

    def finish(self, result: PipelineResult) -> None:
        self.image_processed_at = datetime.now(tz=timezone.utc)
        elapsed = (self.image_processed_at - self.image_received_at).total_seconds() * 1000
        self.total_processing_ms = round(elapsed, 2)
        self.result = result

    def _bbox_dict(self, target: str) -> Optional[Dict[str, Any]]:
        if self.result is None:
            return None
        return (self.result.raw_detections or {}).get(target)

    def to_log_record(self) -> Dict[str, Any]:
        r = self.result

        def _conf(val: Optional[float]) -> Optional[float]:
            return round(val, 4) if val is not None else None

        def _pct(val: Optional[float]) -> Optional[float]:
            return round(val, 2) if val is not None else None

        def _bbox(d: Optional[Dict]) -> Optional[Dict]:
            if d is None:
                return None
            return {
                "x1": round(d["x1"], 4),
                "y1": round(d["y1"], 4),
                "x2": round(d["x2"], 4),
                "y2": round(d["y2"], 4),
            }

        prev_bbox = None
        if self.mouse_previous_bbox is not None:
            mb = self.mouse_previous_bbox
            prev_bbox = {
                "x1": round(mb.x1, 4),
                "y1": round(mb.y1, 4),
                "x2": round(mb.x2, 4),
                "y2": round(mb.y2, 4),
            }

        return {
            # Timing
            "image_received_at":    self.image_received_at.isoformat(),
            "image_processed_at":   self.image_processed_at.isoformat() if self.image_processed_at else None,
            "total_processing_ms":  self.total_processing_ms,
            # Request metadata
            "cage_id":              self.cage_id,
            "filename":             self.filename,
            "image_size_bytes":     self.image_size_bytes,
            # Pipeline outcome
            "success":              r.success if r else False,
            "rejection_reason":     r.rejection_reason if r else None,
            "duplicate_frame":      self.duplicate_frame,
            "uncertain_targets":    r.uncertain_targets if r else [],
            # Water
            "water_pct":            _pct(r.water_pct if r else None),
            "water_label":          r.water_label if r else None,
            "water_confidence":     _conf(r.water_confidence if r else None),
            "water_bbox":           _bbox(self._bbox_dict("water")),
            # Food
            "food_pct":             _pct(r.food_pct if r else None),
            "food_label":           r.food_label if r else None,
            "food_confidence":      _conf(r.food_confidence if r else None),
            "food_bbox":            _bbox(self._bbox_dict("food")),
            # Mouse
            "mouse_present":        r.mouse_present if r else None,
            "mouse_stationary":     r.mouse_stationary if r else None,
            "mouse_confidence":     _conf(r.mouse_confidence if r else None),
            "mouse_bbox":           _bbox(self._bbox_dict("mouse")),
            "mouse_previous_bbox":  prev_bbox,
            "mouse_consecutive_count": self.mouse_consecutive_count,
            # Bedding
            "bedding_condition":    r.bedding_condition  if r else None,
            "bedding_confidence":   _conf(r.bedding_confidence if r else None),
            "bedding_area_pct":     _pct(r.bedding_area_pct if r else None),
            "bedding_bbox":         _bbox(self._bbox_dict("bedding")),
            # Webhook
            "webhook_fired":        self.webhook_fired,
        }


class PipelineEventLogger:
    def __init__(self, config: DotMap) -> None:
        log_cfg = config.logging
        raw_path: str = getattr(log_cfg, "pipeline_log_path", "./logs/pipeline_events.jsonl")
        self._path = Path(raw_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info(f"PipelineEventLogger initialised | path={self._path}")

    def log(self, ctx: PipelineRunContext) -> None:
        record = ctx.to_log_record()
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.error(f"PipelineEventLogger write error: {exc}")
