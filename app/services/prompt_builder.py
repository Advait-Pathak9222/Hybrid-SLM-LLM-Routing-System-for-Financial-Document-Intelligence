"""Prompt construction for financial analysis tasks.

Maps task types to specialised system prompts and wraps user-supplied
financial text into a clean user prompt.
"""

TASK_PROMPTS: dict[str, str] = {
    "summarization": (
        "You are a senior financial analyst. Summarise the following financial "
        "text clearly and concisely. Highlight key figures, trends, and "
        "takeaways. Use bullet points where appropriate."
    ),
    "risk_analysis": (
        "You are a senior financial risk analyst. Identify and evaluate the "
        "key risks present in the following financial text. Categorise risks "
        "by severity (high / medium / low) and suggest potential mitigations."
    ),
    "sentiment_analysis": (
        "You are a financial sentiment analyst. Determine the overall market "
        "sentiment conveyed by the following text. Classify it as bullish, "
        "bearish, or neutral and justify your classification with evidence."
    ),
    "extraction": (
        "You are a financial data extraction specialist. Extract all key "
        "financial metrics, entities, dates, and monetary values from the "
        "following text. Present them in a structured format."
    ),
    "forecasting": (
        "You are a financial forecasting expert. Based on the data and trends "
        "in the following text, provide a reasoned short-term outlook. "
        "State assumptions clearly and quantify projections where possible."
    ),
}

_DEFAULT_SYSTEM_PROMPT = (
    "You are a senior financial analyst. Analyse the following financial text "
    "thoroughly and provide a clear, well-structured response."
)


def build_prompt(task_type: str, financial_text: str) -> tuple[str, str]:
    """Build system and user prompts for a given financial analysis task.

    Args:
        task_type: The category of analysis (e.g. ``"summarization"``).
        financial_text: Raw financial text to analyse.

    Returns:
        A ``(system_prompt, user_prompt)`` tuple ready for model consumption.
    """
    system_prompt = TASK_PROMPTS.get(task_type.lower(), _DEFAULT_SYSTEM_PROMPT)
    user_prompt = f"Financial text for analysis:\n\n{financial_text}"
    return system_prompt, user_prompt
