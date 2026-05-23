"""Extraction service for financial texts.

Provides a thin async function that delegates to the appropriate model tier
and parses the raw response into a structured ExtractionResult.
"""

import re

from app.models.domain import ExtractionResult
from app.router.model_router import ModelTier
from app.services import llm_service, slm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a financial data extraction specialist. Extract all structured "
    "information from the provided financial text. Identify entities (companies, "
    "people, organizations), financial metrics (revenue, EPS, margins, etc.), "
    "dates, and monetary values. Respond using the following structured format:\n\n"
    "ENTITIES:\n- <entity 1>\n- <entity 2>\n\n"
    "METRICS:\n- <metric 1>\n- <metric 2>\n\n"
    "DATES:\n- <date 1>\n- <date 2>\n\n"
    "MONETARY VALUES:\n- <value 1>\n- <value 2>"
)


async def extract(
    financial_text: str, model_tier: ModelTier
) -> ExtractionResult:
    """Extract structured financial data from a text using the specified model tier.

    Args:
        financial_text: The raw financial text to extract data from.
        model_tier: Which model tier (SLM or LLM) to use for generation.

    Returns:
        A structured ExtractionResult with entities, metrics, dates, and monetary values.
    """
    logger.info(
        "Extracting from financial text (%d chars) with model_tier=%s",
        len(financial_text),
        model_tier.value,
    )

    prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{financial_text}"

    if model_tier == ModelTier.SLM:
        raw = await slm_service.generate(prompt, task_type="extraction")
    else:
        raw = await llm_service.generate(prompt, task_type="extraction")

    logger.debug("Raw extraction response length: %d", len(raw))
    return _parse_extraction_response(raw)


def _parse_section_items(text: str, section: str, next_sections: list[str]) -> list[str]:
    """Extract bullet-pointed items from a named section in the raw text.

    Args:
        text: The full raw response text.
        section: The section header to look for (e.g. "ENTITIES").
        next_sections: Headers of subsequent sections used as boundary markers.

    Returns:
        A list of extracted items, or an empty list if the section is missing.
    """
    boundary = "|".join(re.escape(s) for s in next_sections) if next_sections else "$"
    pattern = rf"{re.escape(section)}:\s*\n?(.*?)(?=\n\s*(?:{boundary}):|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    block = match.group(1).strip()
    return [
        line.lstrip("-•* ").strip()
        for line in block.splitlines()
        if line.strip() and line.strip() not in ("-", "•", "*")
    ]


def _parse_extraction_response(raw: str) -> ExtractionResult:
    """Parse a raw model response into an ExtractionResult (best-effort).

    Looks for ENTITIES:, METRICS:, DATES:, and MONETARY VALUES: sections.
    Falls back to putting the entire raw text into entities if parsing fails.
    """
    try:
        entities = _parse_section_items(raw, "ENTITIES", ["METRICS", "DATES", "MONETARY VALUES"])
        metrics = _parse_section_items(raw, "METRICS", ["DATES", "MONETARY VALUES"])
        dates = _parse_section_items(raw, "DATES", ["MONETARY VALUES"])
        monetary_values = _parse_section_items(raw, "MONETARY VALUES", [])

        # If nothing was parsed at all, fall back
        if not any([entities, metrics, dates, monetary_values]):
            return ExtractionResult(entities=[raw.strip()], metrics=[], dates=[], monetary_values=[])

        return ExtractionResult(
            entities=entities,
            metrics=metrics,
            dates=dates,
            monetary_values=monetary_values,
        )
    except Exception:
        logger.warning("Failed to parse extraction response, returning raw in entities")
        return ExtractionResult(entities=[raw.strip()], metrics=[], dates=[], monetary_values=[])
