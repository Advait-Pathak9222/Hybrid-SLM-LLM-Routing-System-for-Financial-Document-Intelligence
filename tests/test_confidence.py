"""Tests for the confidence scoring module."""

import pytest

from app.services.confidence import score_confidence


class TestScoreConfidence:
    """Verify that score_confidence returns appropriate scores for various inputs."""

    def test_good_structured_response_gets_high_confidence(self) -> None:
        """A well-formed, detailed response should score >= 0.7."""
        response = (
            "The company's revenue increased by 15% year-over-year, driven primarily "
            "by strong performance in the cloud services division. Operating margins "
            "expanded by 200 basis points to 28.5%, reflecting improved cost management. "
            "Net income rose to $4.2 billion, representing a 12% increase from the prior "
            "year. The balance sheet remains strong with $18 billion in cash and equivalents."
        )
        score = score_confidence(response, "summarization")
        assert score >= 0.7, f"Expected >= 0.7, got {score}"

    def test_empty_response_gets_very_low_confidence(self) -> None:
        """An empty string should score <= 0.2."""
        score = score_confidence("", "summarization")
        assert score <= 0.4, f"Expected <= 0.4, got {score}"

    def test_uncertain_response_gets_low_confidence(self) -> None:
        """A response full of hedging/uncertainty phrases should score lower."""
        uncertain_response = (
            "I'm not sure about this, but maybe the revenue might have increased. "
            "I don't know the exact figures, and I'm uncertain whether the margins "
            "improved. It's unclear if the company performed well. Perhaps the results "
            "could be positive, but I'm not confident in this assessment."
        )
        score = score_confidence(uncertain_response, "summarization")
        assert score < 0.7, f"Expected < 0.7 for uncertain response, got {score}"

    def test_very_short_response_gets_lower_confidence(self) -> None:
        """A very short response should receive a lower confidence score."""
        score = score_confidence("Revenue up.", "summarization")
        assert score < 0.7, f"Expected < 0.7 for short response, got {score}"

    def test_repetitive_response_gets_lower_confidence(self) -> None:
        """A response with heavy repetition should receive a lower confidence score."""
        repetitive_response = (
            "Revenue increased. Revenue increased. Revenue increased. "
            "Revenue increased. Revenue increased. Revenue increased. "
            "Revenue increased. Revenue increased. Revenue increased."
        )
        score = score_confidence(repetitive_response, "summarization")
        assert score < 0.8, f"Expected < 0.8 for repetitive response, got {score}"
