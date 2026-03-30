from __future__ import annotations

from typing import Dict

from agent_system_a.llm.ollama_client import chat_with_ollama
from agent_system_a.tools.exercise_tools import search_exercises

# ── Prompts ───────────────────────────────────────────────────────────────────

_PARAM_EXTRACTION_PROMPT = """You are a parameter extractor for an exercise search tool.

Extract structured search parameters from the user's message.
Return ONLY a JSON object with these exact keys (omit keys you cannot determine):
  - primary_muscle: one of [chest, back, quadriceps, hamstrings, glutes, shoulders, biceps, triceps, forearms, calves, abdominals, lower back]
  - equipment: one of [dumbbell, barbell, machine, cable, bands, kettlebell, body only, exercise ball, medicine ball, foam roll, e-z curl bar]
  - level: one of [beginner, intermediate, expert]
  - category: one of [strength, stretching, plyometrics, powerlifting, cardio, olympic weightlifting, strongman]

Rules:
- Only include keys you are confident about from the message.
- If the user says "abs" use "abdominals". If they say "legs" or "quads" use "quadriceps".
- If the user says "bodyweight" use "body only".
- If the user says "advanced" use "expert".
- Return raw JSON only. No explanation. No markdown fences.

User message: {message}
"""

_SYNTHESIS_PROMPT = """You are an expert certified personal trainer giving detailed, practical exercise coaching.

The user asked:
{question}

Here are matching exercises retrieved from the database:
{context}

Your task:
- Use ONLY the exercises listed above. Do NOT invent exercises.
- Return the best exercises (up to 10) that match the user's request.
- Start with a 2-3 sentence introduction explaining why these exercises were selected and how they help the user's goal.
- For EACH exercise provide a RICH, DETAILED breakdown:
  * Exercise name (bold it with **)
  * Primary muscle targeted and secondary muscles involved
  * Equipment needed
  * Difficulty level
  * Why this exercise is effective for the user's specific goal
  * Step-by-step execution: how to perform it correctly (3-5 steps)
  * Common mistakes to avoid
  * Sets & reps recommendation based on the user's level
  * A pro coaching tip for maximum results
- End with a brief 2-3 sentence note about progression and how to make the exercises harder over time.
- If the context above contains exercises, present them all. Do NOT add any closing statement or fallback message at the end.
- Only say "No matching exercises were found" if the context is completely empty.
Format each exercise like this:

**[Exercise Name]**
- Muscles: [primary] | Secondary: [if available]
- Equipment: [equipment]
- Level: [level]

Why it works: [2-3 sentences explaining the benefit for the user's goal]

How to perform:
1. [step]
2. [step]
3. [step]

Common mistakes: [what to avoid]
Recommended: [sets x reps, e.g., 3 sets of 10-12 reps]
Pro tip: [coaching insight]

---
"""

# ── Parameter extraction ──────────────────────────────────────────────────────

def _extract_params(message: str) -> dict:
    import json, re

    prompt = _PARAM_EXTRACTION_PROMPT.format(message=message)
    try:
        raw = chat_with_ollama(
            system_prompt="You are a JSON parameter extractor. Return only raw JSON.",
            user_prompt=prompt,
        )
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        params = json.loads(clean)
        allowed = {"primary_muscle", "equipment", "level", "category"}
        return {k: v for k, v in params.items() if k in allowed and v}
    except Exception as exc:
        print(f"[exercise_agent] Param extraction failed: {exc}")
        return {}


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_exercise_query(message: str) -> Dict:
    # Step 1: LLM extracts structured parameters from the message
    params = _extract_params(message)
    print(f"[exercise_agent] Extracted params: {params}")

    # Step 2: Tool filters exercises.json directly
    exercise_context = search_exercises.invoke(params)

    # Step 3: LLM synthesizes a rich, detailed coach-style response
    synthesis_prompt = _SYNTHESIS_PROMPT.format(
        question=message,
        context=exercise_context,
    )
    try:
        final_response = chat_with_ollama(
            system_prompt=(
                "You are an expert certified personal trainer. "
                "Give detailed, practical, encouraging exercise coaching. "
                "Use only exercises from the provided context. "
                "Be specific about form, technique, sets, reps, and progression."
                "Never add 'No matching exercises were found' if exercises were already presented above."
                
            ),
            user_prompt=synthesis_prompt,
        )
    except Exception as exc:
        print(f"[exercise_agent] LLM synthesis failed: {exc}")
        final_response = exercise_context

    return {
        "response": final_response,
        "route": "exercise",
        "data": {
            "extracted_params": params,
            "raw_results": exercise_context,
        },
    }