from __future__ import annotations

from agent_system_a.llm.ollama_client import chat_with_ollama

_COACH_SYSTEM_PROMPT = (
    "You are an expert fitness and nutrition coach. "
    "Answer based only on the provided context. "
    "Do not invent information. "
    "Be concise, practical, and encouraging."
)


def call_llm(prompt: str, timeout: int = 60) -> str:
    """
    Wrapper for specialist agents (exercise, nutrition, program).
    The prompt already contains grounded context.
    """
    return chat_with_ollama(
        system_prompt=_COACH_SYSTEM_PROMPT,
        user_prompt=prompt,
        timeout=timeout,
    )