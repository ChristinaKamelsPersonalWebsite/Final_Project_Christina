"""
ask_nutrition_dietician — LangGraph tool for Agent A

Calls Agent B (CrewAI Nutrition Dietician) via HTTP POST /nutrition-advice.
This replaces the old Nutrition specialist sub-agent inside Agent A.

Agent A's supervisor routes ALL nutrition questions through this tool.
Agent B does the RAG retrieval, LLM reasoning, and safety review independently.

Usage:
    from agent_system_a.tools.nutrition_client import make_nutrition_tool

    # In your graph / agent setup:
    nutrition_tool = make_nutrition_tool(
        session_id=state["session_id"],
        conversation_history=state["messages"],
    )
    tools = [search_exercises, build_workout, analyze_progress, nutrition_tool]
"""

import os
import logging
from typing import Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("nutrition_client")

AGENT_B_URL     = os.getenv("AGENT_B_URL", "http://agent-system-b:8001")
AGENT_B_TIMEOUT = float(os.getenv("AGENT_B_TIMEOUT", "60.0"))  # Ollama can be slow


# ── Input schema (what the LLM sees and fills in) ────────────────────────────

class NutritionQueryInput(BaseModel):
    user_query: str = Field(
        ...,
        description="The user's nutrition question exactly as they asked it.",
    )
    user_context: Optional[dict] = Field(
        default=None,
        description=(
            "Optional user profile to personalize the answer. "
            "Include any known fields: age, weight_kg, height_cm, goal, "
            "dietary_restrictions, tdee, macros, activity_level."
        ),
    )


# ── HTTP call ─────────────────────────────────────────────────────────────────

def _call_agent_b(
    session_id: str,
    user_query: str,
    user_context: Optional[dict],
    conversation_history: list[dict],
) -> str:

    # 🔥 SAFE PAYLOAD (fix 422)
    payload = {
        "session_id": str(session_id),
        "user_query": str(user_query),
        "user_context": user_context if isinstance(user_context, dict) else {},
        "conversation_history": (
            conversation_history if isinstance(conversation_history, list) else []
        ),
    }

    # 🚨 EXTRA SAFETY: ensure each history item is valid dict
    clean_history = []
    for msg in payload["conversation_history"]:
        if isinstance(msg, dict):
            clean_history.append({
                "role": str(msg.get("role", "")),
                "content": str(msg.get("content", "")),
            })

    payload["conversation_history"] = clean_history

    try:
        with httpx.Client(timeout=AGENT_B_TIMEOUT) as client:
            resp = client.post(f"{AGENT_B_URL}/nutrition-advice", json=payload)
            resp.raise_for_status()
            data = resp.json()

        answer     = data.get("answer", "")
        sources    = data.get("sources_used", [])
        follow_ups = data.get("follow_up_suggestions", [])

        parts = [answer]

        if sources:
            parts.append("\n📚 Sources: " + ", ".join(sources))

        if follow_ups:
            fq = "\n".join(f"- {q}" for q in follow_ups)
            parts.append(f"\n💬 Follow-up questions:\n{fq}")

        return "\n".join(parts)

    except httpx.TimeoutException:
        logger.error("Agent B timed out after %.0fs | session=%s", AGENT_B_TIMEOUT, session_id)
        return "Nutrition service is slow right now. Try again."

    except httpx.HTTPStatusError as e:
        logger.error(
            "Agent B HTTP %s | session=%s | response=%s | payload=%s",
            e.response.status_code,
            session_id,
            e.response.text,
            payload,
        )
        return f"Agent B error {e.response.status_code}: {e.response.text}"

    except Exception as e:
        logger.error("Agent B unreachable | session=%s | %s", session_id, str(e))
        return "Nutrition service unavailable."

# ── Tool factory ──────────────────────────────────────────────────────────────

def make_nutrition_tool(
    session_id: str,
    conversation_history: Optional[list] = None,
):
    """
    Factory function that creates the nutrition tool pre-bound to the current session.

    Call this when building the tool list for a session so the tool
    automatically carries session_id and history without the LLM needing to supply them.

    Example:
        tools = [
            search_exercises,
            build_workout,
            analyze_progress,
            make_nutrition_tool(
                session_id=state["session_id"],
                conversation_history=state["messages"],
            ),
        ]
    """

    # Normalize LangChain message objects → plain dicts once
    history_dicts: list[dict] = []
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 3 turns only
            if hasattr(msg, "type") and hasattr(msg, "content"):
                history_dicts.append({"role": msg.type, "content": str(msg.content)})
            elif isinstance(msg, dict):
                history_dicts.append(msg)

    @tool("ask_nutrition_dietician", args_schema=NutritionQueryInput)
    def ask_nutrition_dietician(
        user_query: str,
        user_context: Optional[dict] = None,
    ) -> str:
        """
        Consults the Nutrition Dietician (Agent B) for expert dietary advice.

        Use this tool when the user asks about ANY nutrition topic, including:
          - Macronutrient targets (protein, carbs, fat grams per day)
          - Meal timing — pre-workout and post-workout nutrition
          - Calorie intake, deficits, and surpluses
          - Supplements (creatine, protein powder, BCAAs, vitamins, etc.)
          - Hydration strategies
          - Meal prep and food planning tips
          - Weight loss or muscle gain nutrition plans
          - Specific diets (keto, intermittent fasting, vegan, etc.)
          - Reading nutrition labels or understanding food quality

        Do NOT use for:
          - Exercise selection or technique → use search_exercises
          - Workout program building      → use build_workout
          - Progress and plateau analysis → use analyze_progress
          - TDEE or macro calculations   → use the MCP tools (calculate_tdee, calculate_macros)

        Input:
          user_query    — the nutrition question as the user asked it
          user_context  — optional dict with profile data (weight, goal, restrictions…)

        Output:
          A personalized, evidence-based nutrition answer from the dietician,
          with sources and suggested follow-up questions.
        """
        logger.info(
            "Delegating nutrition query to Agent B | session=%s | query=%.80s",
            session_id, user_query,
        )
        return _call_agent_b(
            session_id=session_id,
            user_query=user_query,
            user_context=user_context,
            conversation_history=history_dicts,
        )

    return ask_nutrition_dietician
