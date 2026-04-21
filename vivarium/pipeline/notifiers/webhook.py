"""
pipeline/notifiers/webhook.py

HTTP Webhook notifier implementation.
Posts a JSON payload to a configured URL. Supports optional HMAC signing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import aiohttp
from dotmap import DotMap
from loguru import logger

from pipeline.notifiers.base import BaseNotifier


class WebhookNotifier(BaseNotifier):
    """Sends alert messages via HTTP webhook."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        webhook_cfg = config.notifiers.webhook
        self._url: str | None = webhook_cfg.url
        self._secret: str | None = webhook_cfg.secret
        self._method: str = str(webhook_cfg.method).upper()
        self._timeout: float = float(webhook_cfg.timeout_seconds)

    async def send(self, message: str, alert_type: str) -> bool:
        """
        POST (or configured method) a JSON payload to the webhook URL.

        Returns:
            True on HTTP 2xx response, False otherwise.
        """
        if not self._url:
            logger.warning("Webhook URL not configured — skipping webhook notification")
            return False

        payload = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": int(time.time()),
        }
        body = json.dumps(payload, ensure_ascii=False)

        headers = {"Content-Type": "application/json"}
        if self._secret:
            sig = hmac.new(
                self._secret.encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Vivarium-Signature"] = f"sha256={sig}"

        try:
            async with aiohttp.ClientSession() as session:
                request_method = getattr(session, self._method.lower())
                async with request_method(
                    self._url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(
                            f"Webhook alert sent | url={self._url} | status={resp.status}"
                        )
                        return True
                    else:
                        logger.error(
                            f"Webhook send failed | status={resp.status} | url={self._url}"
                        )
                        return False
        except Exception as exc:
            logger.error(f"Webhook send error: {exc}")
            return False
