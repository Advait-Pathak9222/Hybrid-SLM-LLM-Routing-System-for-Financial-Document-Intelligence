"""Route a classification decision to the appropriate model tier."""

from enum import Enum

from app.router.task_classifier import RoutingDecision
from app.utils.logger import get_logger

log = get_logger(__name__)


class ModelTier(Enum):
    """Which model tier should serve the request."""

    SLM = "slm"
    LLM = "llm"


def route(decision: RoutingDecision) -> ModelTier:
    """Map a :class:`RoutingDecision` to the target :class:`ModelTier`.

    Args:
        decision: The classification produced by :func:`classify_task`.

    Returns:
        :attr:`ModelTier.SLM` for simple tasks,
        :attr:`ModelTier.LLM` for complex tasks.
    """
    tier = ModelTier.SLM if decision is RoutingDecision.SIMPLE else ModelTier.LLM
    log.info("routing_decision=%s → model_tier=%s", decision.value, tier.value)
    return tier
