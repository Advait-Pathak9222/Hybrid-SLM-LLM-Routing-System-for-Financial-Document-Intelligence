"""Classify incoming task types into routing decisions."""

from enum import Enum

from app.utils.logger import get_logger

log = get_logger(__name__)


class RoutingDecision(Enum):
    """Whether a task should be handled locally (SIMPLE) or in the cloud (COMPLEX)."""

    SIMPLE = "simple"
    COMPLEX = "complex"


# Lookup table: task_type (lowercase) → routing decision.
_TASK_ROUTING: dict[str, RoutingDecision] = {
    # ---- simple (SLM-eligible) ----
    "summarization": RoutingDecision.SIMPLE,
    "extraction": RoutingDecision.SIMPLE,
    "classification": RoutingDecision.SIMPLE,
    "sentiment": RoutingDecision.SIMPLE,
    # ---- complex (LLM-required) ----
    "risk_analysis": RoutingDecision.COMPLEX,
    "trend_analysis": RoutingDecision.COMPLEX,
    "reasoning": RoutingDecision.COMPLEX,
    "multi_step": RoutingDecision.COMPLEX,
    "comparison": RoutingDecision.COMPLEX,
}


def classify_task(task_type: str) -> RoutingDecision:
    """Map a free-form *task_type* string to a :class:`RoutingDecision`.

    Lookup is case-insensitive.  Unknown task types default to
    :attr:`RoutingDecision.COMPLEX` (fail-safe: prefer the more
    capable model when uncertain).

    Args:
        task_type: The analysis task name supplied by the client.

    Returns:
        The corresponding routing decision.
    """
    normalised = task_type.strip().lower()
    decision = _TASK_ROUTING.get(normalised, RoutingDecision.COMPLEX)
    log.info("task_type=%s → %s", normalised, decision.value)
    return decision
