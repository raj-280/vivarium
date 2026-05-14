"""
core/mouse_stationary_tracker.py

Tracks whether a mouse is stationary across consecutive pipeline runs
for a given cage. State is kept per cage_id in memory.

Logic:
    - Each cage has its own last_bbox and consecutive_count.
    - On every run, IoU between current and last bbox is computed.
    - If IoU >= iou_threshold, the mouse is considered in the same place
      and consecutive_count is incremented.
    - If consecutive_count >= consecutive_count config, mouse_stationary=True.
    - If IoU < threshold (mouse moved), counter resets to 1.
    - If mouse not detected in current run, state is NOT updated —
      we can't confirm position so we don't count it either way.

Config block (config.yaml):
    mouse_stationary:
      enabled: true
      iou_threshold: 0.70
      consecutive_count: 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox


@dataclass
class _CageState:
    """Per-cage tracking state."""
    last_bbox: Optional[BoundingBox] = None
    consecutive_count: int = 0


class MouseStationaryTracker:
    """
    Stateful tracker that flags a mouse as stationary if it appears
    in the same position for N consecutive pipeline runs per cage.
    """

    def __init__(self, config: DotMap) -> None:
        self._config = config
        cfg = config.mouse_stationary
        self._enabled: bool = bool(cfg.enabled)
        self._iou_threshold: float = float(cfg.iou_threshold)
        self._consecutive_count: int = int(cfg.consecutive_count)
        self._state: Dict[str, _CageState] = {}
        logger.info(
            f"MouseStationaryTracker initialised | enabled={self._enabled} "
            f"| iou_threshold={self._iou_threshold} "
            f"| consecutive_count={self._consecutive_count}"
        )

    def check(self, cage_id: str, bbox: Optional[BoundingBox]) -> bool:
        if not self._enabled:
            return False

        if bbox is None:
            logger.debug(f"[Tracker] cage={cage_id} | mouse not detected — state unchanged")
            return False

        if cage_id not in self._state:
            self._state[cage_id] = _CageState(last_bbox=bbox, consecutive_count=1)
            logger.debug(f"[Tracker] cage={cage_id} | first observation — state initialised")
            return False

        state = self._state[cage_id]

        if state.last_bbox is None:
            state.last_bbox = bbox
            state.consecutive_count = 1
            return False

        iou = self._compute_iou(state.last_bbox, bbox)
        logger.debug(
            f"[Tracker] cage={cage_id} | iou={iou:.3f} "
            f"threshold={self._iou_threshold} count={state.consecutive_count}"
        )

        if iou >= self._iou_threshold:
            state.consecutive_count += 1
            logger.debug(
                f"[Tracker] cage={cage_id} | same place detected "
                f"| consecutive_count={state.consecutive_count}"
            )
        else:
            state.consecutive_count = 1
            logger.debug(f"[Tracker] cage={cage_id} | mouse moved — count reset")

        state.last_bbox = bbox

        stationary = state.consecutive_count >= self._consecutive_count
        if stationary:
            logger.warning(
                f"[Tracker] cage={cage_id} | MOUSE STATIONARY FLAG | "
                f"consecutive_count={state.consecutive_count} >= {self._consecutive_count} | "
                f"iou={iou:.3f}"
            )
        return stationary

    def get_state(self, cage_id: str) -> Optional[_CageState]:
        """Return the current tracking state for a cage, or None if unseen."""
        return self._state.get(cage_id)

    def reset(self, cage_id: str) -> None:
        """Manually reset state for a cage (e.g. after an alert is handled)."""
        if cage_id in self._state:
            del self._state[cage_id]
            logger.info(f"[Tracker] cage={cage_id} | state reset")

    @staticmethod
    def _compute_iou(a: BoundingBox, b: BoundingBox) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0.0 else 0.0
