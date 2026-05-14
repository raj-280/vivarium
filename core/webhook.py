"""
core/webhook.py

Temporary / lightweight webhook dispatcher.

When a pipeline run completes, this module POSTs the result payload to one
or more configured URLs using httpx (async, with a short timeout).

Design notes
────────────
• "Temporary" means the webhook list lives in config.yaml (no DB, no UI).
  Add / remove URLs by editing config and restarting. This is intentional
  for the first iteration — a proper webhook management endpoint can be
  added later.

• Delivery is fire-and-forget: failures are logged but never raise into the
  pipeline so a slow/dead webhook endpoint cannot block the API response.

• Each POST carries a standard envelope:

    {
      "event":      "pipeline.complete",          // always this value for now
      "api_version": "1",
      "cage_id":    "cage_1",
      "timestamp":  "2025-01-15T10:23:45.123456+00:00",
      "payload":    { ...full pipeline result... }
    }

• An HMAC-SHA256 signature header is added when webhook.secret is set in
  config so receivers can verify authenticity:

    X-Vivarium-Signature: sha256=<hex>

Config block (config.yaml):

    webhook:
      enabled: true
      secret: ""                    # leave blank to skip signing
      timeout_seconds: 5
      urls:
        - https://example.com/hook1
        - https://your-n8n-instance/webhook/vivarium

Usage (called from orchestrator.py):
    from core.webhook import WebhookDispatcher
    dispatcher = WebhookDispatcher(config)            # once at startup
    fired = await dispatcher.dispatch(cage_id, result)   # after every run
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from dotmap import DotMap
from loguru import logger

from core.result import AlertRecord, PipelineResult


_EVENT_TYPE = "pipeline.complete"
_API_VERSION = "1"


class WebhookDispatcher:
    """
    Fire-and-forget async webhook dispatcher.

    Instantiate once at startup, then call `dispatch()` after each
    pipeline run. Returns True if at least one POST succeeded.
    """

    # Exponential backoff retry delays in seconds (3 attempts: 1s, 2s, 4s)
    _RETRY_DELAYS: List[float] = [1.0, 2.0, 4.0]

    def __init__(self, config: DotMap) -> None:
        wh = getattr(config, "webhook", DotMap())
        self._enabled: bool = bool(getattr(wh, "enabled", False))
        self._secret: str = str(getattr(wh, "secret", "") or "")
        self._timeout: float = float(getattr(wh, "timeout_seconds", 5))
        raw_urls = getattr(wh, "urls", []) or []
        self._urls: List[str] = [u for u in raw_urls if u]

        if self._enabled and not self._urls:
            logger.warning("Webhook enabled but no URLs configured — no POSTs will fire")

        logger.info(
            f"WebhookDispatcher initialised | enabled={self._enabled} "
            f"| urls={len(self._urls)} | signed={bool(self._secret)}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._urls)

    def _build_envelope(
        self,
        cage_id: str,
        result: PipelineResult,
        alerts: Optional[List[AlertRecord]] = None,
    ) -> Dict[str, Any]:
        envelope: Dict[str, Any] = {
            "event": _EVENT_TYPE,
            "api_version": _API_VERSION,
            "cage_id": cage_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "payload": result.to_dict(),
        }
        if alerts:
            envelope["alerts"] = [
                {
                    "target": a.target,
                    "alert_type": a.alert_type,
                    "value": a.value,
                    "message": a.message,
                }
                for a in alerts
            ]
        return envelope

    def _sign(self, body: bytes) -> Optional[str]:
        """Return HMAC-SHA256 hex signature, or None if no secret configured."""
        if not self._secret:
            return None
        mac = hmac.new(self._secret.encode(), body, hashlib.sha256)
        return f"sha256={mac.hexdigest()}"

    async def dispatch(
        self,
        cage_id: str,
        result: PipelineResult,
        alerts: Optional[List[AlertRecord]] = None,
    ) -> bool:
        """
        POST result to all configured webhook URLs with exponential backoff retry.

        Attempts up to 3 times per URL (delays: 1s, 2s, 4s).
        Returns True if at least one URL received a 2xx response.
        Never raises — all errors are swallowed and logged.
        """
        if not self.enabled:
            return False

        envelope = self._build_envelope(cage_id, result, alerts)
        body = json.dumps(envelope, ensure_ascii=False).encode()
        signature = self._sign(body)

        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-Vivarium-Signature"] = signature

        any_success = False

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for url in self._urls:
                success = await self._post_with_retry(client, url, body, headers, cage_id)
                if success:
                    any_success = True

        return any_success

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: bytes,
        headers: Dict[str, str],
        cage_id: str,
    ) -> bool:
        """
        Attempt a POST with exponential backoff retry (3 attempts: 1s, 2s, 4s).
        Returns True on first 2xx response, False if all attempts fail.
        """
        for attempt, delay in enumerate(self._RETRY_DELAYS, start=1):
            try:
                resp = await client.post(url, content=body, headers=headers)
                if resp.is_success:
                    logger.info(
                        f"Webhook POST OK | url={url} | status={resp.status_code} "
                        f"| cage={cage_id} | attempt={attempt}"
                    )
                    return True
                else:
                    logger.warning(
                        f"Webhook POST non-2xx | url={url} | status={resp.status_code} "
                        f"| cage={cage_id} | attempt={attempt}/{len(self._RETRY_DELAYS)}"
                    )
            except httpx.TimeoutException:
                logger.warning(
                    f"Webhook timeout | url={url} | cage={cage_id} "
                    f"| attempt={attempt}/{len(self._RETRY_DELAYS)}"
                )
            except httpx.RequestError as exc:
                logger.error(
                    f"Webhook request error | url={url} | cage={cage_id} "
                    f"| attempt={attempt}/{len(self._RETRY_DELAYS)} | {exc}"
                )
            except Exception as exc:
                logger.error(f"Webhook unexpected error | url={url} | {exc}")
                return False  # non-recoverable, don't retry

            if attempt < len(self._RETRY_DELAYS):
                logger.info(
                    f"Webhook retry in {delay:.0f}s | url={url} | cage={cage_id} "
                    f"| next attempt={attempt + 1}/{len(self._RETRY_DELAYS)}"
                )
                await asyncio.sleep(delay)

        logger.error(
            f"Webhook failed after {len(self._RETRY_DELAYS)} attempts | "
            f"url={url} | cage={cage_id}"
        )
        return False
