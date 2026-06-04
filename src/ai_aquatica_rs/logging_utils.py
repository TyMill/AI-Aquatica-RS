"""Logging helpers."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "ai_aquatica_rs", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a single stream handler.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Logging level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
