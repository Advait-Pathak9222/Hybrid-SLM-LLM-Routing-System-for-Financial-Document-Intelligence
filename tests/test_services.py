"""Tests for the domain service modules."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.domain import SummarizationResult, ExtractionResult, AnalysisResult
from app.router.model_router import ModelTier


class TestSummarizer:
    """Tests for the summarization service."""

    @pytest.mark.asyncio
    @patch("app.services.summarizer.slm_service.generate", new_callable=AsyncMock)
    async def test_summarize_slm_returns_result(self, mock_generate: AsyncMock) -> None:
        """SLM summarization should return a SummarizationResult."""
        mock_generate.return_value = (
            "SUMMARY:\nRevenue grew 15% to $4.2B driven by cloud services.\n\n"
            "KEY FIGURES:\n- Revenue: $4.2B\n- Growth: 15%\n\n"
            "SENTIMENT:\nbullish"
        )
        from app.services.summarizer import summarize
        result = await summarize("Revenue grew 15%...", ModelTier.SLM)
        assert isinstance(result, SummarizationResult)
        assert "4.2B" in result.summary or len(result.summary) > 0
        mock_generate.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.services.summarizer.llm_service.generate", new_callable=AsyncMock)
    async def test_summarize_llm_returns_result(self, mock_generate: AsyncMock) -> None:
        """LLM summarization should return a SummarizationResult."""
        mock_generate.return_value = "A concise financial summary of the quarter."
        from app.services.summarizer import summarize
        result = await summarize("Some financial text", ModelTier.LLM)
        assert isinstance(result, SummarizationResult)
        mock_generate.assert_awaited_once()


class TestExtractor:
    """Tests for the extraction service."""

    @pytest.mark.asyncio
    @patch("app.services.extractor.slm_service.generate", new_callable=AsyncMock)
    async def test_extract_returns_result(self, mock_generate: AsyncMock) -> None:
        """Extraction should return an ExtractionResult."""
        mock_generate.return_value = (
            "ENTITIES:\n- Apple Inc.\n- Microsoft\n\n"
            "METRICS:\n- Revenue: $81.8B\n\n"
            "DATES:\n- Q3 2024\n\n"
            "MONETARY VALUES:\n- $81.8 billion"
        )
        from app.services.extractor import extract
        result = await extract("Apple reported Q3 revenue of $81.8B", ModelTier.SLM)
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) > 0
        mock_generate.assert_awaited_once()


class TestAnalyzer:
    """Tests for the financial analysis service."""

    @pytest.mark.asyncio
    @patch("app.services.analyzer.slm_service.generate", new_callable=AsyncMock)
    async def test_analyze_returns_result(self, mock_generate: AsyncMock) -> None:
        """Financial analysis should return an AnalysisResult."""
        mock_generate.return_value = (
            "ANALYSIS:\nThe company shows strong growth with improving margins.\n\n"
            "RISK LEVEL:\nlow\n\n"
            "KEY RISKS:\n- Market competition\n- Regulatory changes\n\n"
            "RECOMMENDATIONS:\n- Maintain current strategy\n- Diversify revenue"
        )
        from app.services.analyzer import analyze_financials
        result = await analyze_financials("Strong quarterly results...", ModelTier.SLM)
        assert isinstance(result, AnalysisResult)
        assert len(result.analysis) > 0
        mock_generate.assert_awaited_once()
