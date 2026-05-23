"""Structured logging configuration for the finance pipeline.

Every log line automatically includes the current ``request_id`` from
:mod:`app.utils.tracing` so that all entries produced during a single
HTTP request can be correlated.
"""

import logging
import sys

from app.utils.tracing import get_request_id

_LOG_FORMAT = (
    "[%(asctime)s] %(levelname)-5s %(name)-30s | req=%(request_id)s | %(message)s"
)
_DEFAULT_LEVEL = logging.INFO


class _RequestIdFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = get_request_id()  # type: ignore[attr-defined]
        return True


# Configure the root logger once at import time.
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_handler.addFilter(_RequestIdFilter())

logging.root.handlers.clear()
logging.root.addHandler(_handler)
logging.root.setLevel(_DEFAULT_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the shared format.

    Args:
        name: Dot-separated logger name, typically ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
