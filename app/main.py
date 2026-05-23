"""Application entry point for the Hybrid SLM-LLM Financial Intelligence Pipeline."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.document_routes import document_router
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
    settings = get_settings()

    # Best-effort Ollama connectivity check
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

    # Initialise ChromaDB document store (if RAG is enabled)
    if settings.rag_enabled:
        try:
            from app.services.document_store import init_document_store

            init_document_store(persist_dir=settings.chroma_persist_dir)
            logger.info("ChromaDB document store initialised | dir=%s", settings.chroma_persist_dir)
        except Exception as exc:
            logger.warning(
                "ChromaDB init failed (%s) — RAG will be unavailable", exc
            )
    else:
        logger.info("RAG is disabled — skipping ChromaDB initialisation")

    # Initialise cache with configured settings
    from app.services.cache import response_cache

    response_cache._max_size = settings.cache_max_size
    response_cache._ttl = settings.cache_ttl_seconds
    logger.info(
        "Response cache configured | max_size=%d  ttl=%ds",
        settings.cache_max_size,
        settings.cache_ttl_seconds,
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
            "automatic confidence-based fallback, response caching, cost "
            "tracking, and RAG-powered document retrieval."
        ),
        version="2.0.0",
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
    application.include_router(document_router)

    # Metrics endpoint
    @application.get("/metrics", tags=["observability"])
    async def get_metrics() -> dict:
        """Return pipeline observability metrics."""
        return pipeline_metrics.summary()

    return application


app = create_app()
