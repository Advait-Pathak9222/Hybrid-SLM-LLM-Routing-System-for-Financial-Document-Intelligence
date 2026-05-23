"""API routes for the financial analysis pipeline."""

from fastapi import APIRouter, HTTPException

from app.schemas.requests import AnalyzeRequest
from app.schemas.responses import AnalyzeResponse
from app.services.orchestrator import analyze
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "healthy"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_financial_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze financial text using the hybrid SLM-LLM pipeline.

    Routes the request to the appropriate model based on task complexity,
    applies confidence scoring, and falls back to the LLM if needed.

    Args:
        request: The analysis request containing financial text and task type.

    Returns:
        AnalyzeResponse with model selection, confidence, latency, and result.

    Raises:
        HTTPException: 500 if an internal error occurs during analysis.
    """
    logger.info(
        "Received /analyze request | task_type=%s | text_length=%d",
        request.task_type,
        len(request.financial_text),
    )
    try:
        response = await analyze(request)
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal error during analysis. Please try again later.",
        ) from exc

    logger.info(
        "Analysis complete | model=%s | routing=%s | confidence=%.2f | latency=%.1fms",
        response.selected_model,
        response.routing_decision,
        response.confidence_score,
        response.latency_ms,
    )
    return response
