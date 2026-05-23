"""Financial analysis service.

Provides a thin async function that delegates to the appropriate model tier
and parses the raw response into a structured AnalysisResult.
"""

import re

from app.models.domain import AnalysisResult
from app.router.model_router import ModelTier
from app.services import llm_service, slm_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an expert financial analyst performing deep analysis on financial "
    "documents. Provide a thorough analysis of the text, assess the overall risk "
    "level, identify key risk factors, and offer actionable recommendations. "
    "Respond using the following structured format:\n\n"
    "ANALYSIS:\n<detailed financial analysis>\n\n"
    "RISK LEVEL:\n<low | medium | high>\n\n"
    "KEY RISKS:\n- <risk 1>\n- <risk 2>\n\n"
    "RECOMMENDATIONS:\n- <recommendation 1>\n- <recommendation 2>"
)


async def analyze_financials(
    financial_text: str, model_tier: ModelTier
) -> AnalysisResult:
    """Perform deep financial analysis on a text using the specified model tier.

    Args:
        financial_text: The raw financial text to analyze.
        model_tier: Which model tier (SLM or LLM) to use for generation.

    Returns:
        A structured AnalysisResult with analysis, risk level, key risks,
        and recommendations.
    """
    logger.info(
        "Analyzing financial text (%d chars) with model_tier=%s",
        len(financial_text),
        model_tier.value,
    )

    prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{financial_text}"

    if model_tier == ModelTier.SLM:
        raw = await slm_service.generate(prompt, task_type="risk_analysis")
    else:
        raw = await llm_service.generate(prompt, task_type="risk_analysis")

    logger.debug("Raw analysis response length: %d", len(raw))
    return _parse_analysis_response(raw)


def _parse_bullet_items(text: str, section: str, next_sections: list[str]) -> list[str]:
    """Extract bullet-pointed items from a named section.

    Args:
        text: The full raw response text.
        section: The section header to look for (e.g. "KEY RISKS").
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


def _parse_analysis_response(raw: str) -> AnalysisResult:
    """Parse a raw model response into an AnalysisResult (best-effort).

    Looks for ANALYSIS:, RISK LEVEL:, KEY RISKS:, and RECOMMENDATIONS: sections.
    Falls back to returning the entire raw text as the analysis if parsing fails.
    """
    try:
        # Extract ANALYSIS section
        analysis_match = re.search(
            r"ANALYSIS:\s*\n?(.*?)(?=\n\s*RISK LEVEL:|\n\s*KEY RISKS:|\n\s*RECOMMENDATIONS:|$)",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        analysis = analysis_match.group(1).strip() if analysis_match else raw.strip()

        # Extract RISK LEVEL
        risk_level = "medium"
        risk_match = re.search(
            r"RISK LEVEL:\s*\n?\s*(low|medium|high)",
            raw,
            re.IGNORECASE,
        )
        if risk_match:
            risk_level = risk_match.group(1).strip().lower()

        # Extract KEY RISKS
        key_risks = _parse_bullet_items(raw, "KEY RISKS", ["RECOMMENDATIONS"])

        # Extract RECOMMENDATIONS
        recommendations = _parse_bullet_items(raw, "RECOMMENDATIONS", [])

        return AnalysisResult(
            analysis=analysis,
            risk_level=risk_level,
            key_risks=key_risks,
            recommendations=recommendations,
        )
    except Exception:
        logger.warning("Failed to parse analysis response, returning raw text")
        return AnalysisResult(
            analysis=raw,
            risk_level="medium",
            key_risks=[],
            recommendations=[],
        )
