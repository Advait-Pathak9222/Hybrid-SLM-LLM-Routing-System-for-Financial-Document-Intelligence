"""Request tracing via context variables.

Provides a per-request correlation ID that propagates automatically
through ``contextvars`` so every log line, service call, and response
can be tied back to a single incoming request.
"""

import uuid
from contextvars import ContextVar

# Holds the active request ID for the current async task.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="no-request")


def generate_request_id() -> str:
    """Create a new UUID4 request identifier."""
    return uuid.uuid4().hex[:12]  # 12-char hex — compact but unique enough


def set_request_id(request_id: str) -> None:
    """Store *request_id* in the current context."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str:
    """Return the current request ID, or ``'no-request'`` outside a request."""
    return _request_id_ctx.get()
