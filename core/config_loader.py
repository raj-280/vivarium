"""
core/config_loader.py

Loads and merges YAML configuration files. Expands ${VAR} and ${VAR:-default}
environment variable patterns. Returns a typed DotMap config object.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotmap import DotMap

from loguru import logger
from dotenv import load_dotenv  # ← import first

load_dotenv()                   # ← then call it
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns from environment."""

    def _replacer(match: re.Match) -> str:
        expr = match.group(1)
        if ":-" in expr:
            var_name, default = expr.split(":-", 1)
            return os.environ.get(var_name.strip(), default.strip())
        return os.environ.get(expr.strip(), match.group(0))  # leave unexpanded if missing

    return _ENV_VAR_PATTERN.sub(_replacer, value)


def _walk_and_expand(obj: Any) -> Any:
    """Recursively walk a parsed YAML structure and expand env vars in strings."""
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(item) for item in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    base_path: str | Path | None = None,
    override_path: str | Path | None = None,
    env: str | None = None,
) -> DotMap:
    """
    Load and merge configuration from YAML files.

    Priority (highest wins): environment override file > env-specific file > base file.

    Args:
        base_path:     Path to config.yaml (defaults to vivarium/config/config.yaml
                       relative to project root).
        override_path: Explicit path to an override YAML (optional).
        env:           Environment name (local | cloud). If provided, attempts to load
                       config.{env}.yaml from the same directory as base_path.

    Returns:
        DotMap: Typed attribute-access configuration object.
    """
    # Resolve base config path
    if base_path is None:
        _here = Path(__file__).resolve().parent.parent  # project root
        base_path = _here / "config" / "config.yaml"

    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as fh:
        merged: dict = yaml.safe_load(fh) or {}

    # Determine env from argument or from the loaded config itself
    active_env = env or merged.get("app", {}).get("env", "local")

    # Load env-specific override (e.g., config.local.yaml)
    env_specific = base_path.parent / f"config.{active_env}.yaml"
    if env_specific.exists():
        with open(env_specific, "r", encoding="utf-8") as fh:
            env_data: dict = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, env_data)

    # Load explicit override if provided
    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path, "r", encoding="utf-8") as fh:
            override_data: dict = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, override_data)

    # Expand all ${ENV_VAR} patterns
    merged = _walk_and_expand(merged)
    # Note: logging is not yet configured at this point (setup_logging runs after),
    # so this debug line will appear with the default loguru format, which is fine.
    logger.debug(f"Config loaded | base={base_path} | env={active_env} | override={override_path}")
    return DotMap(merged, _dynamic=False)


# ---------------------------------------------------------------------------
# Singleton helper — lets the rest of the app call get_config() anywhere.
# ---------------------------------------------------------------------------

_config_instance: DotMap | None = None


def get_config(
    base_path: str | Path | None = None,
    override_path: str | Path | None = None,
    env: str | None = None,
    reload: bool = False,
) -> DotMap:
    """Return (or create) the singleton config instance."""
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = load_config(base_path, override_path, env)
    return _config_instance
