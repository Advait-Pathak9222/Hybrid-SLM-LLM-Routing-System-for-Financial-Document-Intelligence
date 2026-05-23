"""Application entry point for the Hybrid SLM-LLM Financial Intelligence Pipeline."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import RequestTimingMiddleware
from app.api.routes import router
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import pipeline_metrics

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup checks and log lifecycle events."""
    logger.info("Finance pipeline starting up …")

    # Best-effort Ollama connectivity check
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info("Ollama reachable  | models=%s", models)
    except Exception as exc:
        logger.warning(
            "Ollama not reachable (%s) — SLM requests will fail until it's running",
            exc,
        )

    yield
    logger.info("Finance pipeline shutting down …")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI instance with CORS middleware,
        request timing, and routes.
    """
    application = FastAPI(
        title="Hybrid SLM-LLM Financial Intelligence Pipeline",
        description=(
            "Intelligent routing layer that directs financial analysis tasks "
            "to a local SLM (Phi-3 Mini via Ollama) for simple tasks or a "
            "cloud LLM (Llama 3 70B via Groq) for complex ones, with "
            "automatic confidence-based fallback."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware — order matters (outermost first)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestTimingMiddleware)

    # Routes
    application.include_router(router)

    # Metrics endpoint
    @application.get("/metrics", tags=["observability"])
    async def get_metrics() -> dict:
        """Return pipeline observability metrics."""
        return pipeline_metrics.summary()

    return application


app = create_app()

