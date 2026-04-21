"""
core/logger.py

Loguru setup driven entirely by config. Call setup_logging(config) once at
application startup; thereafter all modules use `from loguru import logger`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from dotmap import DotMap


def setup_logging(config: "DotMap") -> None:
    """
    Configure Loguru sinks and format from config.logging settings.

    Supported sinks: stdout | file | loki
    Supported formats: json | text
    """
    # Remove default sink
    logger.remove()

    log_cfg = config.logging
    level: str = config.app.log_level

    fmt_json = (
        '{{"time":"{time:YYYY-MM-DD HH:mm:ss.SSS}", '
        '"level":"{level}", '
        '"module":"{module}", '
        '"function":"{function}", '
        '"line":{line}, '
        '"message":"{message}"}}'
    )
    fmt_text = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    log_format = fmt_json if log_cfg.format == "json" else fmt_text
    serialize = log_cfg.format == "json"

    sink_type: str = log_cfg.sink

    if sink_type == "stdout":
        logger.add(
            sys.stdout,
            format=log_format,
            level=level,
            serialize=serialize,
            colorize=(log_cfg.format != "json"),
        )

    elif sink_type == "file":
        file_path = Path(log_cfg.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(file_path),
            format=log_format,
            level=level,
            serialize=serialize,
            rotation=f"{log_cfg.rotate_mb} MB",
            retention=f"{log_cfg.retain_days} days",
            enqueue=True,
        )

    elif sink_type == "loki":
        # Optional: add loki handler if loki_url is configured
        loki_url = log_cfg.loki_url
        if loki_url:
            try:
                import logging_loki  # type: ignore

                handler = logging_loki.LokiHandler(
                    url=f"{loki_url}/loki/api/v1/push",
                    tags={"application": config.app.name, "env": config.app.env},
                    version="1",
                )
                logger.add(handler, level=level, serialize=True)
            except ImportError:
                logger.warning("logging_loki not installed; falling back to stdout")
                logger.add(sys.stdout, format=log_format, level=level, serialize=serialize)
        else:
            logger.warning("loki_url not set; falling back to stdout")
            logger.add(sys.stdout, format=log_format, level=level, serialize=serialize)

    else:
        # Fallback: always have stdout
        logger.add(sys.stdout, format=log_format, level=level, serialize=serialize)

    logger.info(
        f"Logging initialised | sink={sink_type} | level={level} | format={log_cfg.format}"
    )
