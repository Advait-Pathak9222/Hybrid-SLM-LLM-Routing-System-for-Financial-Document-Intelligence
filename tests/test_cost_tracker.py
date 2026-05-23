"""Tests for the cost tracker."""

import pytest

from app.services.cost_tracker import CostEstimate, estimate_cost, estimate_tokens


class TestEstimateTokens:
    """Token estimation tests."""

    def test_simple_text(self):
        """4 chars ≈ 1 token."""
        assert estimate_tokens("abcd") == 1

    def test_longer_text(self):
        """Longer text scales linearly."""
        text = "a" * 400
        assert estimate_tokens(text) == 100

    def test_empty_text(self):
        """Empty text returns at least 1 token."""
        assert estimate_tokens("") == 1


class TestEstimateCost:
    """Cost estimation tests."""

    def test_local_model_is_free(self):
        """Ollama/local models should always cost $0."""
        result = estimate_cost("phi3:mini", "input text", "output text")
        assert result.estimated_cost_usd == 0.0
        assert result.cost_saved_vs_gpt4o > 0.0

    def test_groq_model_has_cost(self):
        """Groq models should have a non-zero cost."""
        result = estimate_cost(
            "groq/llama3-70b-8192",
            "a" * 4000,  # ~1000 tokens
            "b" * 2000,  # ~500 tokens
        )
        assert result.estimated_cost_usd > 0.0
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500

    def test_savings_vs_gpt4o(self):
        """Savings should be positive for non-GPT-4o models."""
        result = estimate_cost("phi3:mini", "test input", "test output")
        assert result.cost_saved_vs_gpt4o > 0.0

    def test_unknown_model_defaults_free(self):
        """Unknown model identifiers default to zero cost."""
        result = estimate_cost("some/unknown-model", "input", "output")
        assert result.estimated_cost_usd == 0.0

    def test_cost_estimate_dataclass(self):
        """CostEstimate fields are populated correctly."""
        result = estimate_cost("phi3:mini", "hello world!", "response text here")
        assert isinstance(result, CostEstimate)
        assert result.model == "phi3:mini"
        assert result.input_tokens > 0
        assert result.output_tokens > 0
