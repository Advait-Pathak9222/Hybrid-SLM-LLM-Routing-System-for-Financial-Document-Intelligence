"""Orchestrator — the central pipeline coordinator.

Classifies the incoming request, routes it to the appropriate model
tier, evaluates SLM confidence, falls back to the cloud LLM when
the local model is not confident enough, and records pipeline metrics.
"""

from app.config import get_settings
from app.router.model_router import ModelTier, route
from app.router.task_classifier import classify_task
from app.schemas.requests import AnalyzeRequest
from app.schemas.responses import AnalyzeResponse
from app.services import llm_service, slm_service
from app.services.confidence import score_confidence
from app.utils.logger import get_logger
from app.utils.metrics import pipeline_metrics
from app.utils.timer import Timer

logger = get_logger(__name__)


async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full hybrid SLM → LLM analysis pipeline.

    Steps:
        1. Classify the task type into a routing decision.
        2. Route the decision to a model tier (SLM or LLM).
        3. If LLM tier — call the cloud model directly.
        4. If SLM tier — call the local model, score confidence,
           and fall back to the cloud model if confidence is below
           the configured threshold.
        5. Record pipeline metrics for observability.

    Args:
        request: Incoming analysis request with ``financial_text``
                 and ``task_type``.

    Returns:
        An :class:`AnalyzeResponse` containing the final answer,
        routing metadata, confidence score, and latency.
    """
    settings = get_settings()

    async with Timer() as timer:
        # ── 1. Classify & route ─────────────────────────────────
        decision = classify_task(request.task_type)
        tier = route(decision)

        logger.info(
            "Routing  | task=%s  decision=%s  tier=%s",
            request.task_type,
            decision.value,
            tier.value,
        )

        # ── 2. Execute ──────────────────────────────────────────
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

    # ── 3. Record metrics ───────────────────────────────────────
    pipeline_metrics.record(
        routing_decision=routing_decision,
        latency_ms=timer.elapsed_ms,
        confidence=confidence,
    )

    logger.info(
        "Done  | model=%s  routing=%s  confidence=%.2f  latency=%.0fms",
        selected_model,
        routing_decision,
        confidence,
        timer.elapsed_ms,
    )

    return AnalyzeResponse(
        selected_model=selected_model,
        routing_decision=routing_decision,
        confidence_score=round(confidence, 4),
        latency_ms=round(timer.elapsed_ms, 2),
        final_response=response_text,
    )

