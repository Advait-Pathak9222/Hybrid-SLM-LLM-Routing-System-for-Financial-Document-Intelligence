"""Tests for the task classifier module."""

import pytest

from app.router.task_classifier import RoutingDecision, classify_task


class TestClassifyTask:
    """Verify that classify_task maps task types to correct routing decisions."""

    @pytest.mark.parametrize(
        "task_type, expected",
        [
            ("summarization", RoutingDecision.SIMPLE),
            ("extraction", RoutingDecision.SIMPLE),
            ("classification", RoutingDecision.SIMPLE),
            ("sentiment", RoutingDecision.SIMPLE),
        ],
    )
    def test_simple_tasks(self, task_type: str, expected: RoutingDecision) -> None:
        """Simple task types should route to SIMPLE."""
        assert classify_task(task_type) == expected

    @pytest.mark.parametrize(
        "task_type, expected",
        [
            ("risk_analysis", RoutingDecision.COMPLEX),
            ("reasoning", RoutingDecision.COMPLEX),
            ("trend_analysis", RoutingDecision.COMPLEX),
            ("multi_step", RoutingDecision.COMPLEX),
            ("comparison", RoutingDecision.COMPLEX),
        ],
    )
    def test_complex_tasks(self, task_type: str, expected: RoutingDecision) -> None:
        """Complex task types should route to COMPLEX."""
        assert classify_task(task_type) == expected

    def test_unknown_task_defaults_to_complex(self) -> None:
        """Unknown task types should fail-safe to COMPLEX (use the stronger model)."""
        assert classify_task("unknown_task_xyz") == RoutingDecision.COMPLEX

    def test_case_insensitivity(self) -> None:
        """Task classification should be case-insensitive."""
        assert classify_task("SUMMARIZATION") == RoutingDecision.SIMPLE
        assert classify_task("Risk_Analysis") == RoutingDecision.COMPLEX
        assert classify_task("Extraction") == RoutingDecision.SIMPLE
