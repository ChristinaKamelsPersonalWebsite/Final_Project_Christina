from __future__ import annotations

import logging
import requests

from agent_system_a.app.config import get_settings

logger = logging.getLogger(__name__)


def chat_with_ollama(system_prompt: str, user_prompt: str, timeout: int = None) -> str:
    settings = get_settings()
    if timeout is None:
        timeout = max(60, int(settings.http_timeout_seconds))

    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 4096,
        },
    }

    try:
        logger.info("Sending request to Ollama...")

        response = requests.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
            timeout=timeout,  #  timeout here
        )

        response.raise_for_status()

        data = response.json()
        result = (data.get("message") or {}).get("content", "").strip()
        if not result:
            raise ValueError("Empty response content from Ollama.")

        logger.info(f"Ollama response: {result}")

        return result

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out.")
        raise

    except Exception as e:
        logger.error(f"Ollama request failed: {str(e)}")
        raise