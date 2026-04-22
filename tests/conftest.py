# tests/conftest.py
#
# Shared pytest configuration and fixtures.
# All tests load config from the real config/config.yaml + config/config.local.yaml
# so that any change to those files is automatically reflected in tests.

import sys
from pathlib import Path

import pytest
from dotmap import DotMap

# Make the vivarium package importable during tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_real_config() -> DotMap:
    """
    Load the merged config exactly as the app does at runtime:
      config/config.yaml  →  merged with  config/config.local.yaml
    Falls back gracefully if config.local.yaml does not exist.
    """
    from core.config_loader import load_config

    base_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(base_path=base_path)
    return config


@pytest.fixture(scope="session")
def real_config() -> DotMap:
    """
    Session-scoped fixture — config is loaded once per test session.
    Use this in any test that needs values from config.yaml / config.local.yaml.
    """
    return _load_real_config()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return PROJECT_ROOT