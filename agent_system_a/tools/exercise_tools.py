from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_system_a.app.config import get_settings

# ── Known value maps (used by agent for parameter extraction) ─────────────────

KNOWN_MUSCLES = {
    "chest": "chest",
    "back": "back",
    "legs": "quadriceps",
    "quadriceps": "quadriceps",
    "quads": "quadriceps",
    "hamstrings": "hamstrings",
    "glutes": "glutes",
    "shoulders": "shoulders",
    "biceps": "biceps",
    "triceps": "triceps",
    "forearms": "forearms",
    "calves": "calves",
    "abs": "abdominals",
    "abdominals": "abdominals",
    "lower back": "lower back",
}

KNOWN_EQUIPMENT = {
    "dumbbell": "dumbbell",
    "barbell": "barbell",
    "machine": "machine",
    "cable": "cable",
    "bands": "bands",
    "kettlebell": "kettlebell",
    "bodyweight": "body only",
    "body only": "body only",
    "exercise ball": "exercise ball",
    "medicine ball": "medicine ball",
    "foam roll": "foam roll",
    "e-z curl bar": "e-z curl bar",
}

KNOWN_LEVELS = {"beginner", "intermediate", "expert"}
KNOWN_CATEGORIES = {"strength", "stretching", "plyometrics", "powerlifting", "cardio", "olympic weightlifting", "strongman"}

MAX_EXERCISES = 10


# ── Data loading ──────────────────────────────────────────────────────────────

def normalize_text(value: str) -> str:
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


@lru_cache(maxsize=1)
def load_exercises() -> List[Dict]:
    settings = get_settings()
    with open(settings.exercises_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _exercise_to_result(ex: Dict) -> Dict:
    return {
        "name": ex.get("name") or "Unknown Exercise",
        "level": ex.get("level"),
        "equipment": ex.get("equipment"),
        "primaryMuscles": [m for m in ex.get("primaryMuscles", []) if m is not None],
        "secondaryMuscles": [m for m in ex.get("secondaryMuscles", []) if m is not None],
        "instructions": [i for i in ex.get("instructions", []) if i is not None],
        "category": ex.get("category"),
        "mechanic": ex.get("mechanic"),
        "force": ex.get("force"),
    }


# ── Core filter function ──────────────────────────────────────────────────────

def filter_exercises(
    primary_muscle: Optional[str] = None,
    equipment: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = MAX_EXERCISES,
) -> List[Dict]:
    """
    Filter exercises.json directly by structured fields.
    All filters are optional — only provided ones are applied.
    Returns at most `limit` results.
    """
    exercises = load_exercises()
    results: List[Dict] = []

    for ex in exercises:
        if primary_muscle:
            muscles = [str(m).lower() for m in ex.get("primaryMuscles", []) if m is not None]
            if primary_muscle.lower() not in muscles:
                continue

        if equipment:
            if equipment.lower() != str(ex.get("equipment", "")).lower():
                continue

        if level:
            if level.lower() != str(ex.get("level", "")).lower():
                continue

        if category:
            if category.lower() != str(ex.get("category", "")).lower():
                continue

        results.append(_exercise_to_result(ex))

        if len(results) >= limit:
            break

    return results


def _relax_and_retry(
    primary_muscle: Optional[str],
    equipment: Optional[str],
    level: Optional[str],
    category: Optional[str],
    limit: int,
) -> List[Dict]:
    """
    If strict filter returns nothing, progressively relax constraints.
    Priority: muscle > equipment > level > category
    """
    # Drop level first
    if level:
        results = filter_exercises(primary_muscle, equipment, None, category, limit)
        if results:
            return results

    # Drop category
    if category:
        results = filter_exercises(primary_muscle, equipment, level, None, limit)
        if results:
            return results

    # Drop equipment
    if equipment:
        results = filter_exercises(primary_muscle, None, level, category, limit)
        if results:
            return results

    # Muscle only
    if primary_muscle:
        results = filter_exercises(primary_muscle, None, None, None, limit)
        if results:
            return results

    # No filters — return top exercises
    return [_exercise_to_result(ex) for ex in load_exercises()[:limit]]


def _format_exercises_for_llm(exercises: List[Dict]) -> str:
    """
    Formats filtered exercises into a clean readable block for the LLM to synthesize from.
    """
    if not exercises:
        return "No matching exercises found."

    lines = [f"Found {len(exercises)} matching exercise(s):\n"]
    for i, ex in enumerate(exercises, 1):
        muscles = ", ".join(ex.get("primaryMuscles", [])) or "unknown"
        secondary = ", ".join(ex.get("secondaryMuscles", [])) or "none"
        instructions = ex.get("instructions", [])
        steps = " ".join(f"({j+1}) {s}" for j, s in enumerate(instructions[:3]))

        lines.append(
            f"{i}. {ex['name']}\n"
            f"   Level: {ex.get('level', 'unknown')} | "
            f"Equipment: {ex.get('equipment', 'unknown')} | "
            f"Category: {ex.get('category', 'unknown')}\n"
            f"   Primary muscles: {muscles}\n"
            f"   Secondary muscles: {secondary}\n"
            f"   How to do it: {steps}\n"
        )
    return "\n".join(lines)


# ── Input schema for the LangGraph tool ──────────────────────────────────────

class ExerciseSearchInput(BaseModel):
    primary_muscle: Optional[str] = Field(
        default=None,
        description=(
            "Target muscle group. Must be one of: chest, back, quadriceps, hamstrings, "
            "glutes, shoulders, biceps, triceps, forearms, calves, abdominals, lower back. "
            "Use canonical names (e.g. 'abdominals' not 'abs', 'quadriceps' not 'quads')."
        ),
    )
    equipment: Optional[str] = Field(
        default=None,
        description=(
            "Equipment available. Must be one of: dumbbell, barbell, machine, cable, "
            "bands, kettlebell, body only, exercise ball, medicine ball, foam roll, e-z curl bar. "
            "Use 'body only' for bodyweight exercises."
        ),
    )
    level: Optional[str] = Field(
        default=None,
        description="Difficulty level: beginner, intermediate, or expert.",
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Exercise category. One of: strength, stretching, plyometrics, "
            "powerlifting, cardio, olympic weightlifting, strongman."
        ),
    )


# ── LangGraph tool ────────────────────────────────────────────────────────────

@tool("search_exercises", args_schema=ExerciseSearchInput)
def search_exercises(
    primary_muscle: Optional[str] = None,
    equipment: Optional[str] = None,
    level: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """
    Searches the exercise database by structured filters and returns up to 10 matching exercises.

    Use this tool when the user asks for exercises, workouts, or movements targeting
    specific muscle groups, equipment, difficulty levels, or exercise categories.

    The LLM should extract parameters from the user's message and pass them as filters.
    All parameters are optional — only pass the ones the user actually specified.
    Returns a formatted list of exercises with instructions ready for the LLM to use.

    Examples:
      - "chest exercises with dumbbells" → primary_muscle="chest", equipment="dumbbell"
      - "beginner leg workout" → primary_muscle="quadriceps", level="beginner"
      - "bodyweight abs" → primary_muscle="abdominals", equipment="body only"
      - "show me shoulder exercises" → primary_muscle="shoulders"
    """
    # Strict filter first
    results = filter_exercises(
        primary_muscle=primary_muscle,
        equipment=equipment,
        level=level,
        category=category,
        limit=MAX_EXERCISES,
    )

    # Relax constraints progressively if nothing found
    if not results:
        results = _relax_and_retry(primary_muscle, equipment, level, category, MAX_EXERCISES)

    return _format_exercises_for_llm(results)


# ── Legacy helpers (kept for program_builder_agent compatibility) ─────────────

def get_exercise_by_name(name: str) -> Optional[Dict]:
    target = normalize_text(name)
    for ex in load_exercises():
        if normalize_text(ex.get("name", "")) == target:
            return _exercise_to_result(ex)
    return None


def search_exercises_by_keywords(query: str, limit: int = 10) -> List[Dict]:
    """Legacy keyword search — kept for backwards compatibility."""
    muscle = next((v for k, v in KNOWN_MUSCLES.items() if k in query.lower()), None)
    equipment = next((v for k, v in KNOWN_EQUIPMENT.items() if k in query.lower()), None)
    level = next((l for l in KNOWN_LEVELS if l in query.lower()), None)

    results = filter_exercises(primary_muscle=muscle, equipment=equipment, level=level, limit=limit)
    if results:
        return results
    return _relax_and_retry(muscle, equipment, level, None, limit)
