"""
core/level_labeler.py

Converts a numeric level percentage into a human-readable label
based on threshold bands defined in config.yaml under level_labels.

Config structure:
    level_labels:
      water:
        - max: 20
          label: "Critical"
        - max: 40
          label: "Low"
        - max: 70
          label: "OK"
        - max: 100
          label: "Full"
      food:
        - max: 20
          label: "Critical"
        ...

Logic:
    Thresholds are walked in order. First band where pct <= max wins.
    If pct is None or no band matches, returns "Unknown".
"""

from __future__ import annotations

from typing import List, Optional

from dotmap import DotMap
from loguru import logger


def get_label(pct: Optional[float], bands: list) -> str:
    """
    Map a percentage to a label using ordered threshold bands.

    Args:
        pct:   The percentage value (0-100), or None if measurement failed.
        bands: List of dicts with 'max' and 'label' keys, in ascending order.

    Returns:
        Human-readable label string.
    """
    if pct is None:
        return "Unknown"

    for band in bands:
        if pct <= float(band["max"]):
            return str(band["label"])

    # Fallback — shouldn't happen if bands cover 0-100
    logger.warning(f"No label band matched for pct={pct} — returning 'Unknown'")
    return "Unknown"


def label_water(pct: Optional[float], config: DotMap) -> str:
    """Get label for water level from config."""
    try:
        bands = [dict(b) for b in config.level_labels.water]
        return get_label(pct, bands)
    except Exception as exc:
        logger.warning(f"Failed to resolve water label: {exc}")
        return "Unknown"


def label_food(pct: Optional[float], config: DotMap) -> str:
    """Get label for food level from config."""
    try:
        bands = [dict(b) for b in config.level_labels.food]
        return get_label(pct, bands)
    except Exception as exc:
        logger.warning(f"Failed to resolve food label: {exc}")
        return "Unknown"
