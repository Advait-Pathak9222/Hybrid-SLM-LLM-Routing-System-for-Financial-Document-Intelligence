"""Tests for the model router module."""

from app.router.model_router import ModelTier, route
from app.router.task_classifier import RoutingDecision


class TestRoute:
    """Verify that route maps routing decisions to the correct model tier."""

    def test_simple_routes_to_slm(self) -> None:
        """SIMPLE tasks should be routed to the SLM (local model)."""
        assert route(RoutingDecision.SIMPLE) == ModelTier.SLM

    def test_complex_routes_to_llm(self) -> None:
        """COMPLEX tasks should be routed to the LLM (cloud model)."""
        assert route(RoutingDecision.COMPLEX) == ModelTier.LLM
