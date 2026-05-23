"""Small Language Model (SLM) service — Ollama backend.

Calls a locally-hosted Ollama instance to generate responses for
financial analysis tasks that are classified as *simple*.
"""

import httpx

from app.config import get_settings
from app.services.prompt_builder import build_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def generate(financial_text: str, task_type: str) -> str:
    """Generate a response from the local SLM via Ollama.

    Args:
        financial_text: The raw financial text to analyse.
        task_type: The category of analysis task.

    Returns:
        The model's response text, or an empty string on timeout /
        connection failure (the confidence scorer will handle this).
    """
    settings = get_settings()
    system_prompt, user_prompt = build_prompt(task_type, financial_text)

    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
    }

    logger.info(
        "SLM call  | model=%s  task=%s  text_len=%d",
        settings.ollama_model,
        task_type,
        len(financial_text),
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.slm_timeout_sec)
        ) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            result: str = resp.json().get("response", "")
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        logger.warning("SLM unavailable (%s), returning empty response", exc)
        return ""
    except httpx.HTTPStatusError as exc:
        logger.warning("SLM HTTP error %s, returning empty response", exc.response.status_code)
        return ""

    logger.info("SLM done  | response_len=%d", len(result))
    return result
