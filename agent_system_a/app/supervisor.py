from __future__ import annotations

import logging
import re
from typing import Dict, List

from agent_system_a.llm.ollama_client import chat_with_ollama

logger = logging.getLogger(__name__)


SUPERVISOR_PROMPT = """
You are the supervisor of a fitness coaching multi-agent system.

Your job is to decide which specialist agent should act next.

Available actions:
- exercise
- nutrition
- program
- progress
- finish

Rules:
- Return exactly one word only.
- Do not explain.
- Do not add punctuation.

Decision logic:
- If the user asks for specific exercises (e.g. "give me chest exercises", "what exercises for legs"), choose: exercise
- If the user asks about nutrition, diet, protein, calories, macros, food, eating, choose: nutrition
- If the user asks for a workout plan, program, schedule, or routine, choose: program
- If the user shares workout logs and asks about progress, plateaus, improvement, PRs, or getting stronger, choose: progress
- If enough information is already gathered, choose: finish
"""


def _build_context(message: str, completed_agents: List[str], history: List[Dict] = None) -> str:
    context = ""
    if history:
        recent = history[-4:]  # last 2 exchanges
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content'][:100]}\n"
        context = f"Recent conversation:\n{context}\n"
    
    return (
        f"{context}"
        f"Current user request: {message}\n"
        f"Completed agents: {completed_agents}\n\n"
        "What should happen next?\n"
        "Return exactly one of:\n"
        "exercise\n"
        "nutrition\n"
        "program\n"
        "progress\n"
        "finish"
    )


def _normalize_decision(raw: str) -> str:
    text = raw.strip().lower()

    for ch in [".", ",", ":", ";", "!", "?", "\"", "'"]:
        text = text.replace(ch, "")

    text = text.splitlines()[0].strip()

    # Extract first valid keyword using word boundaries
    for keyword in ["exercise", "nutrition", "program", "progress", "finish"]:
        if re.search(r'\b' + keyword + r'\b', text):
            return keyword

    return "finish"


def _keyword_route(message: str) -> str | None:
    """Fast keyword-based routing before calling LLM."""
    text = message.lower()

    progress_keywords = [
        "plateau", "plateauing", "getting stronger", "am i improving",
        "analyze my progress", "my progress", "workout log", "bench press log",
        "squat log", "deadlift log", "reps 3 sets", "sets on 202",
        "making progress", "check my logs", "my logs",
    ]
    for kw in progress_keywords:
        if kw in text:
            return "progress"

    nutrition_keywords = [
        "protein", "calorie", "calories", "macro", "macros", "diet",
        "eat", "eating", "food", "nutrition", "carb", "fat intake", "meal",
    ]
    for kw in nutrition_keywords:
        if kw in text:
            return "nutrition"

    exercise_keywords = [
        "exercise", "exercises", "movement", "workout for", "what to do for",
        "give me", "show me", "best exercise",
    ]
    for kw in exercise_keywords:
        if kw in text:
            return "exercise"

    program_keywords = [
        "program", "plan", "routine", "schedule", "week program",
        "day program", "build me", "create a", "make me a",
    ]
    for kw in program_keywords:
        if kw in text:
            return "program"
    conversational = ["thanks", "thank you", "bye", "goodbye", "exit", "ok", "okay", "great", "cool", "gd bye"]
    for kw in conversational:
        if text.strip() == kw or text.strip().startswith(kw):
            return "conversational"        

    return None


def decide_next_agent(
    message: str,
    completed_agents: List[str] | None = None,
    partial_results: Dict | None = None,
) -> str:
    completed_agents = completed_agents or []
    partial_results = partial_results or {}

    logger.info(f"Supervisor received message: {message}")
    logger.info(f"Completed agents: {completed_agents}")

    # If already completed a single-agent task, finish
    if any(a in completed_agents for a in ["program", "progress"]):
        return "finish"
    if "exercise" in completed_agents and "nutrition" not in completed_agents:
        # Check if nutrition was also requested
        if not any(kw in message.lower() for kw in ["nutrition", "protein", "calories", "diet", "eat"]):
            return "finish"

    # Fast keyword routing first (no LLM needed)
    keyword_decision = _keyword_route(message)
    if keyword_decision and keyword_decision not in completed_agents:
        logger.info(f"Keyword routing decision: {keyword_decision}")
        return keyword_decision

    # Fall back to LLM routing
    try:
        user_prompt = _build_context(message, completed_agents, partial_results.get("messages", []))
        logger.info("===== USER PROMPT SENT TO OLLAMA =====")
        logger.info(user_prompt)
        logger.info("======================================")

        raw = chat_with_ollama(
            system_prompt=SUPERVISOR_PROMPT,
            user_prompt=user_prompt
        )

        logger.info(f"Raw supervisor output: {raw}")

        decision = _normalize_decision(raw)
        logger.info(f"Supervisor next decision: {decision}")

        return decision

    except Exception as e:
        logger.error(f"Supervisor failed: {str(e)}")
        # Use keyword decision as fallback
        return keyword_decision or "finish"


def route_message(message: str) -> str:
    return decide_next_agent(message=message, completed_agents=[], partial_results={})
