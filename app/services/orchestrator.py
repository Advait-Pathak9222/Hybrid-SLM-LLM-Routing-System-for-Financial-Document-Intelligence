"""Orchestrator — the central pipeline coordinator.

Classifies the incoming request, checks the response cache, routes it
to the appropriate model tier, evaluates SLM confidence, falls back to
the cloud LLM when the local model is not confident enough, tracks
cost, and records pipeline metrics.
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.router.model_router import ModelTier, route
from app.router.task_classifier import classify_task
from app.schemas.requests import AnalyzeRequest
from app.schemas.responses import AnalyzeResponse
from app.services import llm_service, slm_service
from app.services.cache import CacheEntry, ResponseCache, response_cache
from app.services.confidence import score_confidence
from app.services.cost_tracker import CostEstimate, estimate_cost
from app.utils.logger import get_logger
from app.utils.metrics import pipeline_metrics
from app.utils.timer import Timer
from app.utils.tracing import get_request_id

logger = get_logger(__name__)


async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full hybrid SLM → LLM analysis pipeline.

    Steps:
        1. Check response cache for an identical prior request.
        2. Retrieve context via RAG (if enabled and requested).
        3. Classify the task type into a routing decision.
        4. Route the decision to a model tier (SLM or LLM).
        5. If LLM tier — call the cloud model directly.
        6. If SLM tier — call the local model, score confidence,
           and fall back to the cloud model if confidence is below
           the configured threshold.
        7. Estimate per-request cost and token usage.
        8. Store the result in cache for future identical requests.
        9. Record pipeline metrics for observability.

    Args:
        request: Incoming analysis request with ``financial_text``
                 and ``task_type``.

    Returns:
        An :class:`AnalyzeResponse` containing the final answer,
        routing metadata, confidence score, latency, cost, and
        tracing information.
    """
    settings = get_settings()
    request_id = get_request_id()

    # ── 0. Cache lookup ─────────────────────────────────────────
    cache_key = ResponseCache.make_key(request.task_type, request.financial_text)
    cached = response_cache.get(cache_key)

    if cached is not None:
        pipeline_metrics.record(
            routing_decision=cached.routing_decision,
            latency_ms=0.0,
            confidence=cached.confidence,
            cache_hit=True,
            tokens_used=cached.tokens_used,
            estimated_cost_usd=0.0,  # cached — no inference cost
        )
        return AnalyzeResponse(
            selected_model=cached.model,
            routing_decision=cached.routing_decision,
            confidence_score=round(cached.confidence, 4),
            latency_ms=0.0,
            final_response=cached.response,
            request_id=request_id,
            cache_hit=True,
            tokens_used=cached.tokens_used,
            estimated_cost_usd=0.0,
            rag_sources_used=0,
        )

    async with Timer() as timer:
        # ── 1. RAG retrieval (Phase 3 — wired in later) ─────────
        rag_sources_used = 0
        context_chunks: list[str] = []
        if getattr(request, "use_rag", False) and settings.rag_enabled:
            try:
                from app.services.document_store import document_store

                top_k = getattr(request, "rag_top_k", settings.rag_default_top_k)
                results = document_store.search(request.financial_text, top_k=top_k)
                context_chunks = [r.text for r in results]
                rag_sources_used = len(context_chunks)
                if rag_sources_used:
                    logger.info(
                        "RAG retrieved %d chunks for context", rag_sources_used
                    )
            except Exception as exc:
                logger.warning("RAG retrieval failed (%s), proceeding without", exc)

        # ── 2. Classify & route ─────────────────────────────────
        decision = classify_task(request.task_type)
        tier = route(decision)

        logger.info(
            "Routing  | task=%s  decision=%s  tier=%s",
            request.task_type,
            decision.value,
            tier.value,
        )

        # ── 3. Build prompt input ───────────────────────────────
        # The prompt_builder will weave in context_chunks if present.
        from app.services.prompt_builder import build_prompt

        system_prompt, user_prompt = build_prompt(
            request.task_type, request.financial_text, context_chunks=context_chunks or None
        )
        full_input_text = f"{system_prompt}\n{user_prompt}"

        # ── 4. Execute inference ────────────────────────────────
        if tier == ModelTier.LLM:
            response_text = await llm_service.generate(
                request.financial_text, request.task_type
            )
            selected_model = f"groq/{settings.groq_model}"
            routing_decision = "llm"
            confidence = 1.0

        else:  # SLM tier
            response_text = await slm_service.generate(
                request.financial_text, request.task_type
            )
            confidence = score_confidence(response_text, request.task_type)

            if confidence >= settings.confidence_threshold:
                selected_model = settings.ollama_model
                routing_decision = "slm"
            else:
                logger.info(
                    "SLM confidence %.2f < threshold %.2f — falling back to LLM",
                    confidence,
                    settings.confidence_threshold,
                )
                response_text = await llm_service.generate(
                    request.financial_text, request.task_type
                )
                selected_model = f"groq/{settings.groq_model}"
                routing_decision = "slm→llm_fallback"

    # ── 5. Cost estimation ──────────────────────────────────────
    cost: CostEstimate = estimate_cost(
        model=selected_model,
        input_text=full_input_text,
        output_text=response_text,
    )

    # ── 6. Store in cache ───────────────────────────────────────
    response_cache.put(
        cache_key,
        CacheEntry(
            response=response_text,
            model=selected_model,
            confidence=confidence,
            routing_decision=routing_decision,
            tokens_used=cost.total_tokens,
            estimated_cost_usd=cost.estimated_cost_usd,
            created_at=time.time(),
        ),
    )

    # ── 7. Record metrics ───────────────────────────────────────
    pipeline_metrics.record(
        routing_decision=routing_decision,
        latency_ms=timer.elapsed_ms,
        confidence=confidence,
        cache_hit=False,
        tokens_used=cost.total_tokens,
        estimated_cost_usd=cost.estimated_cost_usd,
        cost_saved_vs_gpt4o=cost.cost_saved_vs_gpt4o,
    )

    logger.info(
        "Done  | model=%s  routing=%s  confidence=%.2f  latency=%.0fms  "
        "tokens=%d  cost=$%.6f  rag_chunks=%d",
        selected_model,
        routing_decision,
        confidence,
        timer.elapsed_ms,
        cost.total_tokens,
        cost.estimated_cost_usd,
        rag_sources_used,
    )

    return AnalyzeResponse(
        selected_model=selected_model,
        routing_decision=routing_decision,
        confidence_score=round(confidence, 4),
        latency_ms=round(timer.elapsed_ms, 2),
        final_response=response_text,
        request_id=request_id,
        cache_hit=False,
        tokens_used=cost.total_tokens,
        estimated_cost_usd=cost.estimated_cost_usd,
        rag_sources_used=rag_sources_used,
    )
