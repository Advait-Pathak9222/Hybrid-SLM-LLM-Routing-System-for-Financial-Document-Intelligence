"""Large Language Model (LLM) service — Groq backend via LiteLLM.

Calls a cloud-hosted LLM for complex financial analysis tasks or as a
fallback when the local SLM produces low-confidence output.

Includes automatic retry with exponential back-off for transient errors.
"""

import asyncio

import litellm

from app.config import get_settings
from app.services.prompt_builder import build_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
BASE_DELAY_SEC = 1.0


async def _call_with_retries(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    timeout: int,
) -> str:
    """Call LiteLLM with exponential back-off retries.

    Args:
        model: The LiteLLM model identifier (e.g. ``"groq/llama3-70b-8192"``).
        messages: Chat-style messages list.
        api_key: Provider API key.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The model's response text.

    Raises:
        Exception: Re-raised after all retry attempts are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content  # type: ignore[union-attr]

        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    "LLM attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.exception(
                    "LLM call failed after %d attempts  | model=%s",
                    MAX_RETRIES,
                    model,
                )

    raise last_exc  # type: ignore[misc]


async def generate(financial_text: str, task_type: str) -> str:
    """Generate a response from the cloud LLM via Groq.

    Retries up to :data:`MAX_RETRIES` times with exponential back-off
    on transient failures.

    Args:
        financial_text: The raw financial text to analyse.
        task_type: The category of analysis task.

    Returns:
        The model's response text.

    Raises:
        Exception: Re-raised after logging on all retries exhausted.
    """
    settings = get_settings()
    system_prompt, user_prompt = build_prompt(task_type, financial_text)

    model = f"groq/{settings.groq_model}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "LLM call  | model=%s  task=%s  text_len=%d",
        model,
        task_type,
        len(financial_text),
    )

    result = await _call_with_retries(
        model=model,
        messages=messages,
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_sec,
    )

    logger.info("LLM done  | response_len=%d", len(result))
    return result

