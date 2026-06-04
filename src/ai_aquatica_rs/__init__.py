"""Top-level public API for AI-Aquatica-RS."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import AppConfig, load_config
from .exceptions import AIAquaticaError, ConfigError, DataValidationError
from .logging_utils import get_logger

try:
    __version__ = version("ai-aquatica-rs")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "AIAquaticaError",
    "AppConfig",
    "ConfigError",
    "DataValidationError",
    "__version__",
    "get_logger",
    "load_config",
]
