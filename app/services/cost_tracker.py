"""Per-request cost estimation based on token counts.

Provides rough token counting and a cost lookup table so that every
response can report ``estimated_cost_usd`` and ``cost_saved_vs_gpt4o``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── cost tables (USD per 1 million tokens) ─────────────────────────
# fmt: off
_COST_TABLE: dict[str, tuple[float, float]] = {
    # model_prefix           : (input_per_1M, output_per_1M)
    "phi3":                    (0.00,   0.00),     # local — free
    "ollama":                  (0.00,   0.00),     # any ollama model
    "groq/llama3-70b-8192":    (0.59,   0.79),     # Groq paid tier
    "groq/llama-3.1-70b":      (0.59,   0.79),
    "groq/llama3-8b-8192":     (0.05,   0.08),
    "groq":                    (0.59,   0.79),     # fallback for any groq model
}

_GPT4O_INPUT_PER_1M = 5.00
_GPT4O_OUTPUT_PER_1M = 15.00
# fmt: on


@dataclass(frozen=True)
class CostEstimate:
    """Result of a per-request cost estimation."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_saved_vs_gpt4o: float
    model: str


def estimate_tokens(text: str) -> int:
    """Rough token count using the ``len / 4`` heuristic.

    This is deliberately simple — a production system would use
    ``tiktoken`` or the provider's tokenizer.  The 4-char rule is
    accurate to within ~15 % for English text.
    """
    return max(1, len(text) // 4)


def _lookup_costs(model: str) -> tuple[float, float]:
    """Return ``(input_per_1M, output_per_1M)`` for *model*."""
    model_lower = model.lower()
    # Exact match first.
    if model_lower in _COST_TABLE:
        return _COST_TABLE[model_lower]
    # Prefix match (e.g. "groq/llama3-70b-8192" matches "groq").
    for prefix, costs in _COST_TABLE.items():
        if model_lower.startswith(prefix):
            return costs
    # Unknown model — assume free (safe default).
    return (0.0, 0.0)


def estimate_cost(
    model: str,
    input_text: str,
    output_text: str,
) -> CostEstimate:
    """Estimate the cost of a single inference call.

    Args:
        model: Model identifier (e.g. ``"phi3:mini"`` or ``"groq/llama3-70b-8192"``).
        input_text: The prompt / user text sent to the model.
        output_text: The response text returned by the model.

    Returns:
        A :class:`CostEstimate` with token counts, USD cost, and GPT-4o savings.
    """
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    total_tokens = input_tokens + output_tokens

    input_cost_per_1m, output_cost_per_1m = _lookup_costs(model)
    cost = (input_tokens * input_cost_per_1m + output_tokens * output_cost_per_1m) / 1_000_000

    gpt4o_cost = (
        input_tokens * _GPT4O_INPUT_PER_1M + output_tokens * _GPT4O_OUTPUT_PER_1M
    ) / 1_000_000
    saved = gpt4o_cost - cost

    logger.info(
        "Cost estimate | model=%s  tokens=%d  cost=$%.6f  saved_vs_gpt4o=$%.6f",
        model,
        total_tokens,
        cost,
        saved,
    )

    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(cost, 8),
        cost_saved_vs_gpt4o=round(saved, 8),
        model=model,
    )
