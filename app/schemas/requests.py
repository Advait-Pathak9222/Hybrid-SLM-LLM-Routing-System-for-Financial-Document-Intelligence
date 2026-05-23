"""Request schemas for the finance pipeline API."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Payload sent by clients to request financial-text analysis."""

    financial_text: str = Field(
        ...,
        min_length=1,
        description="Financial text to analyze",
    )
    task_type: str = Field(
        ...,
        min_length=1,
        description="Type of analysis task",
    )

    # --- RAG options ---
    use_rag: bool = Field(
        default=False,
        description="If true, retrieve relevant context from ingested documents before analysis",
    )
    rag_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve from the document store",
    )
