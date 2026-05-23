"""Response schemas for the finance pipeline API."""

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    """Payload returned to clients after financial-text analysis."""

    selected_model: str = Field(
        ...,
        description="Model used for inference",
    )
    routing_decision: str = Field(
        ...,
        description="Routing path taken: slm, llm, or slm→llm_fallback",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response",
    )
    latency_ms: float = Field(
        ...,
        description="End-to-end latency in milliseconds",
    )
    final_response: str = Field(
        ...,
        description="The analysis result",
    )
