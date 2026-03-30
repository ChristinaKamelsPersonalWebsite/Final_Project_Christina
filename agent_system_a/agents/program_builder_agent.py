from __future__ import annotations

from typing import Dict, List, Optional

from agent_system_a.app.schemas import ProgramBuilderParams
from agent_system_a.llm.ollama_client import chat_with_ollama
from agent_system_a.tools.program_tools import build_weekly_plan


# ── LLM-based parameter extraction ───────────────────────────────────────────

_PARAM_EXTRACTION_PROMPT = """You are a parameter extractor for a workout program builder.

Extract structured parameters from the user's message.
Return ONLY a JSON object with these exact keys (omit keys you cannot determine):
  - goal: one of [strength, hypertrophy, fat loss, general fitness, endurance]
  - days_per_week: integer between 2 and 6
  - duration_weeks: integer, typically 4, 6, 8, or 12
  - experience_level: one of [beginner, intermediate, expert]
  - equipment: list of strings from [dumbbell, barbell, bodyweight, gym, home, kettlebell, bands]

Rules:
- If the user says "build muscle" or "gain muscle" use "hypertrophy".
- If the user says "lose weight" or "fat loss" use "fat loss".
- If the user says "get stronger" use "strength".
- If the user says "advanced" use "expert".
- If the user mentions "home" with no equipment, use equipment: ["home"].
- If the user mentions "gym" use equipment: ["gym"].
- If no duration is mentioned, default duration_weeks to 4.
- If no days are mentioned, default days_per_week to 3.
- If no level is mentioned, default experience_level to "beginner".
- If no goal is mentioned, default goal to "general fitness".
- Return raw JSON only. No explanation. No markdown fences.

User message: {message}
"""


def _extract_program_params(message: str) -> ProgramBuilderParams:
    import json
    import re

    try:
        raw = chat_with_ollama(
            system_prompt="You are a JSON parameter extractor. Return only raw JSON.",
            user_prompt=_PARAM_EXTRACTION_PROMPT.format(message=message),
        )
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        params = json.loads(clean)

        return ProgramBuilderParams(
            goal=params.get("goal", "general fitness"),
            days_per_week=int(params.get("days_per_week", 3)),
            duration_weeks=int(params.get("duration_weeks", 4)),
            experience_level=params.get("experience_level", "beginner"),
            equipment=params.get("equipment") or None,
        )

    except Exception as exc:
        print(f"[program_builder_agent] Param extraction failed, using defaults: {exc}")
        return ProgramBuilderParams(
            goal="general fitness",
            days_per_week=3,
            duration_weeks=4,
            experience_level="beginner",
            equipment=None,
        )


# ── Plan formatting ────────────────────────────────────────────────────────────

def _clean_exercise_names(day: Dict) -> List[str]:
    names: List[str] = []
    seen = set()
    for ex in day.get("exercises", []):
        name = ex.get("name")
        if not name:
            continue
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _plan_has_too_much_repetition(plan: Dict) -> bool:
    all_names: List[str] = []
    for day in plan.get("schedule", []):
        for ex in day.get("exercises", []):
            name = ex.get("name")
            if name:
                all_names.append(name)

    if not all_names:
        return True

    unique_count = len(set(all_names))
    total_count = len(all_names)
    return unique_count < max(4, total_count // 3)


def _format_plan(plan: Dict) -> str:
    schedule = plan.get("schedule", [])
    num_days = len(schedule)
    lines = [
        f"Program: {plan.get('plan_name', 'Custom Plan')}",
        f"Goal: {plan.get('goal', 'general fitness')}",
        f"Days per week: {plan.get('days_per_week', 'unknown')}",
        f"Total training days: {num_days} (DO NOT ADD MORE DAYS)",
        f"Level: {plan.get('level', 'unknown')}",
        "Schedule:",
    ]
    for day in schedule:
        focus = day.get("focus", "Workout")
        day_name = day.get("day", "Day")

        ex_details = []
        for ex in day.get("exercises", []):
            name = ex.get("name", "")
            sets = ex.get("sets", "")
            reps = ex.get("reps", "")
            muscle = ex.get("primary_muscle", "")
            equipment = ex.get("equipment", "")
            if sets and reps:
                detail = f"{name} ({sets}x{reps})"
            else:
                detail = name
            if muscle:
                detail += f" [targets: {muscle}]"
            if equipment:
                detail += f" [equipment: {equipment}]"
            ex_details.append(detail)

        lines.append(f"\n{day_name} - {focus}:")
        for ex in ex_details:
            lines.append(f"  • {ex}")

    lines.append(f"\n--- END OF PLAN: {num_days} DAYS TOTAL. DO NOT ADD ANY MORE DAYS. ---")
    return "\n".join(lines)


# ── LLM synthesis ─────────────────────────────────────────────────────────────

_PROGRAM_SYNTHESIS_PROMPT = """You are an expert certified strength and conditioning coach creating a detailed workout program.

User request:
{question}

Structured workout plan from the database:
{plan}

CRITICAL RULES:
- Show ONLY the exact days listed in the structured plan above.
- Count the days in the plan. If there are 3 days, show EXACTLY 3 days. NEVER add Day 4, 5, 6, 7, 8, 9, 10, or 11.
- Do NOT invent extra days or exercises not in the plan.
- Stop after presenting all days in the plan.
- Do NOT add a "Daily Routine Example" or weekly schedule at the end. Stop after "What to Expect".

Your task:
- Start with a 3-4 sentence introduction explaining the program's philosophy, why it matches the user's goal, and what results they can expect.
- For EACH training day provide:
  * Day name and focus (e.g., "Day 1 - Upper Body Strength")
  * For each exercise:
    - Exercise name
    - Sets x Reps (keep exactly as in the plan)
    - Which muscles it targets
    - A brief technique tip (1-2 sentences)
  * Rest time between sets recommendation
  * Estimated workout duration
- After the schedule, add a "Progressive Overload" section explaining how to increase difficulty each week.
- Add a "Recovery & Nutrition Tips" section with 3-4 practical tips.
- Add a "What to Expect" section describing results timeline.
- Keep all exercise names, sets, reps exactly as provided. Do NOT invent new exercises.
- Be encouraging, specific, and practical.
"""


def _synthesize_with_llm(question: str, formatted_plan: str) -> str:
    system_prompt = (
        "You are an expert certified strength and conditioning coach. "
        "CRITICAL: Only show the exact number of days provided in the plan. "
        "If the plan has 3 days, show ONLY 3 days. Never add extra days. "
        "Stop presenting days as soon as you have covered all days in the plan. "
        "Keep all exercise names, sets, and reps exactly as provided in the plan."
    )
    user_prompt = _PROGRAM_SYNTHESIS_PROMPT.format(
        question=question,
        plan=formatted_plan,
    )
    try:
        result = chat_with_ollama(system_prompt, user_prompt)
        print(f"[program_builder_agent] LLM result: {result[:100]}")
        return result
    except Exception as exc:
        print(f"[program_builder_agent] LLM synthesis failed, using raw plan: {exc}")
        return formatted_plan


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_program_query(message: str) -> Dict:
    # Step 1: LLM extracts structured params from user message
    params = _extract_program_params(message)
    print(f"[program_builder_agent] Extracted params: {params.model_dump()}")

    # Step 2: Filter workout_programs.json and substitute exercises
    plan = build_weekly_plan(params)

    # Step 3: Validate plan quality
    warning = None
    if _plan_has_too_much_repetition(plan):
        warning = (
            "Note: this plan is structurally valid, "
            "but exercise variety may still need refinement."
        )

    # Step 4: Format for LLM synthesis with more detail
    structured_text = _format_plan(plan)
    if warning:
        structured_text += f"\n\n{warning}"

    # Step 5: LLM rewrites into rich coach-style response
    final_response = _synthesize_with_llm(message, structured_text)

    return {
        "response": final_response,
        "route": "program",
        "data": {
            "params": params.model_dump(),
            "plan": plan,
            "warning": warning,
        },
    }