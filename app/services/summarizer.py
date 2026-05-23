"""Summarization service for financial texts.

Provides a thin async function that delegates to the appropriate model tier
and parses the raw response into a structured SummarizationResult.
"""

import re

from app.models.domain import SummarizationResult
from app.router.model_router import ModelTier
from app.services import llm_service, slm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Summarize the provided financial text "
    "concisely, highlighting the most important points, key financial figures, "
    "and overall market sentiment. Respond using the following structured format:\n\n"
    "SUMMARY:\n<concise summary>\n\n"
    "KEY FIGURES:\n- <figure 1>\n- <figure 2>\n\n"
    "SENTIMENT:\n<bullish | bearish | neutral>"
)


async def summarize(
    financial_text: str, model_tier: ModelTier
) -> SummarizationResult:
    """Summarize a financial text using the specified model tier.

    Args:
        financial_text: The raw financial text to summarize.
        model_tier: Which model tier (SLM or LLM) to use for generation.

    Returns:
        A structured SummarizationResult with summary, key figures, and sentiment.
    """
    logger.info(
        "Summarizing financial text (%d chars) with model_tier=%s",
        len(financial_text),
        model_tier.value,
    )

    prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{financial_text}"

    if model_tier == ModelTier.SLM:
        raw = await slm_service.generate(prompt, task_type="summarization")
    else:
        raw = await llm_service.generate(prompt, task_type="summarization")

    logger.debug("Raw summarization response length: %d", len(raw))
    return _parse_summary_response(raw)


def _parse_summary_response(raw: str) -> SummarizationResult:
    """Parse a raw model response into a SummarizationResult (best-effort).

    Looks for SUMMARY:, KEY FIGURES:, and SENTIMENT: sections. Falls back to
    returning the entire raw text as the summary if parsing fails.
    """
    try:
        # Extract SUMMARY section
        summary_match = re.search(
            r"SUMMARY:\s*\n?(.*?)(?=\n\s*KEY FIGURES:|\n\s*SENTIMENT:|$)",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        summary = summary_match.group(1).strip() if summary_match else raw.strip()

        # Extract KEY FIGURES section
        key_figures: list[str] = []
        figures_match = re.search(
            r"KEY FIGURES:\s*\n?(.*?)(?=\n\s*SENTIMENT:|$)",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if figures_match:
            figures_text = figures_match.group(1).strip()
            key_figures = [
                line.lstrip("-•* ").strip()
                for line in figures_text.splitlines()
                if line.strip() and line.strip() not in ("-", "•", "*")
            ]

        # Extract SENTIMENT section
        sentiment = "neutral"
        sentiment_match = re.search(
            r"SENTIMENT:\s*\n?\s*(bullish|bearish|neutral)",
            raw,
            re.IGNORECASE,
        )
        if sentiment_match:
            sentiment = sentiment_match.group(1).strip().lower()

        return SummarizationResult(
            summary=summary,
            key_figures=key_figures,
            sentiment=sentiment,
        )
    except Exception:
        logger.warning("Failed to parse summary response, returning raw text")
        return SummarizationResult(summary=raw, key_figures=[], sentiment="neutral")
