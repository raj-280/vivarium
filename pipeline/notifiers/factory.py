"""
pipeline/notifiers/factory.py

Returns the list of enabled BaseNotifier implementations from
notifiers.enabled config key.

Callers must never import notifier implementations directly.
"""

from __future__ import annotations

import importlib
from typing import List

from dotmap import DotMap

from pipeline.notifiers.base import BaseNotifier


class ConfigurationError(Exception):
    """Raised when an unsupported notifier name is found in config."""


_REGISTRY: dict[str, str] = {
    "telegram": "pipeline.notifiers.telegram.TelegramNotifier",
    "email": "pipeline.notifiers.email_notifier.EmailNotifier",
    "webhook": "pipeline.notifiers.webhook.WebhookNotifier",
}


class NotifierFactory:
    """Factory that instantiates all enabled notifiers from config."""

    @staticmethod
    def create_all(config: DotMap) -> List[BaseNotifier]:
        """
        Instantiate and return all notifiers listed in notifiers.enabled.

        Args:
            config: Full DotMap configuration object.

        Returns:
            List of instantiated BaseNotifier subclass instances.

        Raises:
            ConfigurationError: If any enabled notifier name is unknown.
        """
        enabled: List[str] = list(config.notifiers.enabled)
        notifiers: List[BaseNotifier] = []

        for name in enabled:
            name_lower = name.lower().strip()
            if name_lower not in _REGISTRY:
                supported = ", ".join(_REGISTRY.keys())
                raise ConfigurationError(
                    f"Unknown notifier '{name_lower}' in notifiers.enabled. "
                    f"Supported: {supported}. "
                    f"Check config.yaml → notifiers.enabled"
                )
            module_path, class_name = _REGISTRY[name_lower].rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            notifiers.append(cls(config))

        return notifiers
