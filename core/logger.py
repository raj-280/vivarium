"""
core/logger.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from dotmap import DotMap


def setup_logging(config: "DotMap") -> None:
    logger.remove()

    log_cfg = config.logging
    level: str = config.app.log_level
    is_local: bool = config.app.env == "local"

    fmt = (
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    shared = dict(
        format=fmt,
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    sink_type: str = log_cfg.sink

    if sink_type == "file":
        file_path = Path(log_cfg.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            **shared,
            colorize=False,
            rotation=f"{log_cfg.rotate_mb} MB",
            retention=f"{log_cfg.retain_days} days",
            enqueue=True,
        )

    elif sink_type == "loki":
        loki_url = log_cfg.loki_url
        if loki_url:
            try:
                import logging_loki  # type: ignore
                handler = logging_loki.LokiHandler(
                    url=f"{loki_url}/loki/api/v1/push",
                    tags={"application": config.app.name, "env": config.app.env},
                    version="1",
                )
                logger.add(handler, level=level, backtrace=False, diagnose=False)
            except ImportError:
                logger.add(sys.stdout, **shared)
        else:
            logger.add(sys.stdout, **shared)

    else:
        logger.add(sys.stdout, **shared)