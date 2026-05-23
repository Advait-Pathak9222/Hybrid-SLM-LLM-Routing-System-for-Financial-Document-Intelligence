"""Response schemas for the finance pipeline API."""

from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    """Payload returned to clients after financial-text analysis."""

    # --- Core fields (existing) ---
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

    # --- Tracing ---
    request_id: str = Field(
        default="",
        description="Correlation ID for distributed tracing",
    )

    # --- Caching ---
    cache_hit: bool = Field(
        default=False,
        description="Whether the response was served from cache",
    )

    # --- Cost tracking ---
    tokens_used: int = Field(
        default=0,
        description="Estimated total token count (input + output)",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        description="Estimated cost in USD for this request",
    )

    # --- RAG ---
    rag_sources_used: int = Field(
        default=0,
        description="Number of retrieved context chunks injected into the prompt",
    )
