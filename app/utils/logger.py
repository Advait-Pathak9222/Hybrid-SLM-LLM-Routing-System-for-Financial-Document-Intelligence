"""Structured logging configuration for the finance pipeline."""

import logging
import sys

_LOG_FORMAT = "[%(asctime)s] %(levelname)-5s %(name)-30s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

# Configure the root logger once at import time.
logging.basicConfig(
    format=_LOG_FORMAT,
    level=_DEFAULT_LEVEL,
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the shared format.

    Args:
        name: Dot-separated logger name, typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
