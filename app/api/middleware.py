"""Middleware for request timing and distributed tracing."""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logger import get_logger
from app.utils.tracing import generate_request_id, set_request_id

logger = get_logger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time and X-Request-Id headers; log request latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # --- Request tracing ---
        # Honour an incoming header (e.g. from an API gateway) or generate one.
        incoming_id = request.headers.get("X-Request-Id")
        request_id = incoming_id or generate_request_id()
        set_request_id(request_id)

        # --- Timing ---
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # --- Response headers ---
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Request-Id"] = request_id

        logger.info(
            "HTTP %s %s | %d | %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
