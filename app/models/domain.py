"""Domain result models for financial pipeline services."""

from pydantic import BaseModel, Field


class SummarizationResult(BaseModel):
    """Structured output from the summarization service."""

    summary: str = Field(..., description="Concise summary of the financial text")
    key_figures: list[str] = Field(
        default_factory=list, description="Key financial figures mentioned"
    )
    sentiment: str = Field(
        default="neutral",
        description="Overall sentiment: bullish, bearish, or neutral",
    )


class ExtractionResult(BaseModel):
    """Structured output from the extraction service."""

    entities: list[str] = Field(
        default_factory=list, description="Financial entities found"
    )
    metrics: list[str] = Field(
        default_factory=list, description="Financial metrics extracted"
    )
    dates: list[str] = Field(default_factory=list, description="Dates mentioned")
    monetary_values: list[str] = Field(
        default_factory=list, description="Monetary values found"
    )


class AnalysisResult(BaseModel):
    """Structured output from the financial analysis service."""

    analysis: str = Field(..., description="Detailed financial analysis")
    risk_level: str = Field(
        default="medium", description="Overall risk: low, medium, high"
    )
    key_risks: list[str] = Field(
        default_factory=list, description="Identified risk factors"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Action recommendations"
    )
