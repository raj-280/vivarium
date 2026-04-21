"""
pipeline/notifiers/email_notifier.py

SMTP email notifier implementation.
Reads all email settings from config. Supports STARTTLS.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotmap import DotMap
from loguru import logger

from pipeline.notifiers.base import BaseNotifier


class EmailNotifier(BaseNotifier):
    """Sends alert messages via SMTP email."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        email_cfg = config.notifiers.email
        self._smtp_host: str = email_cfg.smtp_host
        self._smtp_port: int = int(email_cfg.smtp_port)
        self._use_tls: bool = bool(email_cfg.use_tls)
        self._from_addr: str = email_cfg["from"]
        self._to_addr: str = email_cfg.to
        self._subject_prefix: str = email_cfg.subject_prefix

    async def send(self, message: str, alert_type: str) -> bool:
        """
        Send email alert via SMTP. Runs blocking SMTP in a thread executor.

        Returns:
            True on success, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._send_sync, message, alert_type
        )

    def _send_sync(self, message: str, alert_type: str) -> bool:
        """Blocking SMTP send (called from thread executor)."""
        subject = f"{self._subject_prefix} {alert_type.replace('_', ' ').title()}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = self._to_addr

        plain_part = MIMEText(message, "plain", "utf-8")
        msg.attach(plain_part)

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=15) as smtp:
                if self._use_tls:
                    smtp.starttls()
                smtp.sendmail(self._from_addr, [self._to_addr], msg.as_string())
            logger.info(f"Email alert sent | to={self._to_addr} | alert_type={alert_type}")
            return True
        except Exception as exc:
            logger.error(f"Email send error: {exc}")
            return False
