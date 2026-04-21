"""
pipeline/notifiers/telegram.py

Telegram Bot API notifier implementation.
Reads bot_token and chat_id from config (which expands them from env vars).
"""

from __future__ import annotations

import aiohttp
from dotmap import DotMap
from loguru import logger

from pipeline.notifiers.base import BaseNotifier


class TelegramNotifier(BaseNotifier):
    """Sends alert messages via Telegram Bot API."""

    _API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        tg_cfg = config.notifiers.telegram
        self._bot_token: str = tg_cfg.bot_token
        self._chat_id: str = str(tg_cfg.chat_id)
        self._parse_mode: str = tg_cfg.parse_mode

    async def send(self, message: str, alert_type: str) -> bool:
        """
        Send message to Telegram chat.

        Returns:
            True on success (HTTP 200 + ok=True), False otherwise.
        """
        url = self._API_BASE.format(token=self._bot_token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": self._parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    body = await resp.json()
                    if resp.status == 200 and body.get("ok"):
                        logger.info(f"Telegram alert sent | alert_type={alert_type}")
                        return True
                    else:
                        logger.error(
                            f"Telegram send failed | status={resp.status} | body={body}"
                        )
                        return False
        except Exception as exc:
            logger.error(f"Telegram send error: {exc}")
            return False
