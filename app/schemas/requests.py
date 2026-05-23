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
