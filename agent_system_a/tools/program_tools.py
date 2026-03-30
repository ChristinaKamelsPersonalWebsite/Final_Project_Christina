from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from agent_system_a.app.config import get_settings
from agent_system_a.app.schemas import ProgramBuilderParams


LEVEL_ORDER: Dict[str, int] = {"beginner": 0, "intermediate": 1, "expert": 2}


def normalize_level(level: Optional[str]) -> str:
    if not level:
        return "intermediate"
    v = level.strip().lower()
    if v in {"beginner", "intermediate", "expert"}:
        return v
    if v in {"advanced", "pro"}:
        return "expert"
    return "intermediate"


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(value: str) -> Set[str]:
    stop = {
        "and", "or", "the", "a", "an", "with", "to", "for", "build",
        "improve", "increase", "strength", "muscle", "fitness", "program",
        "plan", "get", "your", "week", "weeks", "at", "of", "in",
    }
    words = set(normalize_text(value).split())
    return {w for w in words if w and w not in stop}


def token_overlap(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(max(len(ta), len(tb)))


def normalize_equipment_input(
    equipment: Optional[Union[str, Sequence[str]]],
) -> Set[str]:
    if equipment is None:
        return set()
    if isinstance(equipment, str):
        raw_parts = re.split(r"[,;/|]+", equipment)
        parts = [p.strip() for p in raw_parts if p.strip()]
    else:
        parts = [str(p).strip() for p in equipment if str(p).strip()]

    normalized: Set[str] = set()
    for p in parts:
        v = p.lower()
        if v in {"gym", "facility"}:
            normalized.add("gym")
            continue
        if v in {"home", "minimal", "minimal equipment"}:
            normalized.add("home")
            continue
        if v in {"bodyweight", "body weight", "bodyweight only", "body"}:
            normalized.add("bodyweight")
            normalized.add("body only")
            continue
        if "dumbbell" in v:
            normalized.add("dumbbell")
        elif "kettlebell" in v:
            normalized.add("kettlebell")
        elif v in {"barbell", "e-z curl bar"}:
            normalized.add(v)
        else:
            normalized.add(v)

    return normalized


def classify_equipment_group(user_equipment: Set[str]) -> str:
    if not user_equipment:
        return "gym"
    eq = set(user_equipment)
    if eq.intersection({"gym", "machine", "cable", "barbell", "e-z curl bar"}):
        return "gym"
    if eq.intersection({"dumbbell", "dumbbells", "kettlebell", "kettlebells", "bands", "ball", "exercise ball", "medicine ball", "other"}):
        return "home"
    return "bodyweight"


def level_distance(a: str, b: str) -> int:
    ia, ib = LEVEL_ORDER.get(a, 1), LEVEL_ORDER.get(b, 1)
    return abs(ia - ib)


def adapt_reps_string(
    reps: Union[str, int, float],
    from_level: str,
    to_level: str,
) -> str:
    reps_str = str(reps).strip()
    delta = LEVEL_ORDER.get(to_level, 1) - LEVEL_ORDER.get(from_level, 1)
    if delta == 0:
        return reps_str

    if "sec" in reps_str:
        amounts = [int(x) for x in re.findall(r"\d+", reps_str)]
        if not amounts:
            return reps_str
        shift = -2 if delta > 0 else 2
        new = [max(1, x + shift) for x in amounts]
        if len(new) >= 2 and "-" in reps_str:
            return f"{new[0]}-{new[1]} sec" + ("/side" if "/side" in reps_str else "")
        return f"{new[0]} sec" + ("/side" if "/side" in reps_str else "")

    m_range = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", reps_str)
    if m_range:
        low, high = int(m_range.group(1)), int(m_range.group(2))
        shift = -1 if delta > 0 else 1
        return f"{max(1, low + shift)}-{max(1, high + shift)}"

    m_single = re.fullmatch(r"\s*(\d+)\s*", reps_str)
    if m_single:
        val = int(m_single.group(1))
        shift = -1 if delta > 0 else 1
        return str(max(1, val + shift))

    m_per = re.fullmatch(r"\s*(\d+)\s*/\s*([a-zA-Z]+)\s*", reps_str)
    if m_per:
        val = int(m_per.group(1))
        unit = m_per.group(2)
        shift = -1 if delta > 0 else 1
        return f"{max(1, val + shift)}/{unit}"

    nums = re.findall(r"\d+", reps_str)
    if nums:
        shift = -1 if delta > 0 else 1
        updated = reps_str
        for n in nums:
            updated = updated.replace(n, str(max(1, int(n) + shift)), 1)
        return updated

    return reps_str


def normalize_equipment_token(token: str) -> str:
    v = token.lower().strip()
    if v in {"dumbbells", "dumbbell"}:
        return "dumbbell"
    if v in {"kettlebells", "kettlebell"}:
        return "kettlebell"
    return v


def equipment_compatible(
    exercise_equipment: Optional[str],
    user_equipment: Set[str],
    *,
    user_group: str,
) -> bool:
    if not exercise_equipment:
        return True
    ex = normalize_equipment_token(exercise_equipment)
    if user_group == "gym":
        return True
    if user_group == "bodyweight":
        return ex in {"body only", "foam roll", "other"}
    if user_group == "home":
        return ex in {
            "dumbbell", "kettlebell", "bands", "medicine ball",
            "exercise ball", "foam roll", "other", "body only",
        } or ex in user_equipment
    return ex in user_equipment


def parse_exercise_template_name(name: str) -> str:
    v = name.strip()
    v = re.sub(r"\(.*?\)", "", v).strip()
    parts = re.split(r"\s+or\s+", v, flags=re.IGNORECASE)
    return parts[0].strip()


def select_best_template(
    templates: List[Dict[str, Any]],
    *,
    goal: str,
    level: str,
    equipment_group: str,
    days_per_week: int,
) -> Dict[str, Any]:
    best: Tuple[float, Dict[str, Any]] = (-1e9, templates[0])
    for t in templates:
        t_goal = str(t.get("goal", ""))
        t_level = normalize_level(t.get("level"))
        t_days = int(t.get("days_per_week", 0) or 0)
        t_equip = str(t.get("equipment", "")).lower().strip()

        goal_score = token_overlap(goal, t_goal) * 6.0
        level_score = -float(level_distance(level, t_level)) * 2.0
        days_score = 4.0 if t_days == days_per_week else -5.0 * abs(t_days - days_per_week)

        equip_score = 0.0
        if t_equip == "gym":
            equip_score = 2.0 if equipment_group == "gym" else 0.5
        elif t_equip == "home":
            equip_score = 2.0 if equipment_group == "home" else 0.5
        elif t_equip == "bodyweight":
            equip_score = 2.0 if equipment_group == "bodyweight" else 0.5

        total = goal_score + level_score + days_score + equip_score
        if total > best[0]:
            best = (total, t)
    return best[1]


@lru_cache(maxsize=1)
def _load_workout_programs() -> List[Dict[str, Any]]:
    s = get_settings()
    with open(s.workout_programs_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_exercises() -> List[Dict[str, Any]]:
    s = get_settings()
    with open(s.exercises_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_exercise_index(
    exercises: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    by_exact: Dict[str, Dict[str, Any]] = {}
    for ex in exercises:
        name = ex.get("name")
        if not name:
            continue
        by_exact[str(name).strip().lower()] = ex
    return by_exact, exercises


@lru_cache(maxsize=1)
def _exercise_cache() -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    exercises = _load_exercises()
    return _build_exercise_index(exercises)


def find_exercise(template_exercise_name: str) -> Optional[Dict[str, Any]]:
    by_exact, _all = _exercise_cache()
    key = template_exercise_name.strip().lower()
    if key in by_exact:
        return by_exact[key]
    alt = parse_exercise_template_name(template_exercise_name).lower()
    return by_exact.get(alt)


def exercise_match_score(
    *,
    candidate: Dict[str, Any],
    original: Optional[Dict[str, Any]],
    user_level: str,
    user_equipment: Set[str],
    user_group: str,
) -> float:
    if candidate is None:
        return -1e9

    c_level = normalize_level(candidate.get("level"))
    c_equip = candidate.get("equipment")
    c_category = candidate.get("category")

    if not equipment_compatible(c_equip, user_equipment, user_group=user_group):
        return -1e9
    if level_distance(c_level, user_level) > 2:
        return -1e9

    score = 0.0
    score += (2.5 - float(level_distance(c_level, user_level))) * 2.0

    if original:
        o_category = original.get("category")
        if o_category and c_category and o_category == c_category:
            score += 3.0

        o_level = normalize_level(original.get("level"))
        score += (2.0 - float(level_distance(c_level, o_level))) * 0.6

        o_muscles = set(original.get("primaryMuscles") or [])
        c_muscles = set(candidate.get("primaryMuscles") or [])
        if o_muscles and c_muscles:
            overlap = len(o_muscles & c_muscles)
            score += float(overlap) * 0.8

    return score


def choose_replacement_exercise(
    *,
    original_template_name: str,
    original_exercise: Optional[Dict[str, Any]],
    user_level: str,
    user_equipment: Set[str],
    user_group: str,
    already_used: Set[str],  # NEW: track used exercises for diversity
) -> Dict[str, Any]:
    """
    Choose a replacement exercise avoiding repeats already used in this plan.
    Falls back to allowing repeats only if no unique exercise is available.
    """
    _, all_exercises = _exercise_cache()

    original_candidate = find_exercise(original_template_name)
    if original_candidate and original_candidate != original_exercise:
        original_exercise = original_candidate

    # Score all candidates, penalize already-used exercises
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for cand in all_exercises:
        cand_name = (cand.get("name") or "").strip().lower()
        if not cand_name:
            continue

        s = exercise_match_score(
            candidate=cand,
            original=original_exercise,
            user_level=user_level,
            user_equipment=user_equipment,
            user_group=user_group,
        )

        # Heavy penalty for already-used exercises to enforce diversity
        if cand_name in already_used:
            s -= 50.0

        scored.append((s, cand))

    if not scored:
        if original_exercise is not None:
            return original_exercise
        raise RuntimeError("No exercise candidates found.")

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cand = scored[0]

    # If best score is still valid (> -1e8 means it's not completely incompatible)
    if best_score > -1e8:
        return best_cand

    if original_exercise is not None:
        return original_exercise

    for cand in all_exercises:
        if equipment_compatible(cand.get("equipment"), user_equipment, user_group=user_group):
            return cand

    raise RuntimeError("No exercise candidates found for the provided constraints.")


def adapt_weekly_plan_from_template(
    template: Dict[str, Any],
    *,
    params: ProgramBuilderParams,
) -> Dict[str, Any]:
    user_level = normalize_level(params.experience_level)
    user_equipment = normalize_equipment_input(params.equipment)
    user_group = classify_equipment_group(user_equipment)

    plan: Dict[str, Any] = {
        "plan_name": f"{params.goal}-{user_level}-{params.days_per_week}d",
        "goal": params.goal,
        "level": user_level,
        "equipment_group": user_group,
        "days_per_week": params.days_per_week,
        "duration_weeks": params.duration_weeks,
        "schedule": [],
        "source_template": template.get("program_name"),
        "adaptations": [],
    }

    template_schedule = template.get("schedule", [])

    # Track all exercises used across entire plan for diversity
    plan_used_exercises: Set[str] = set()

    def adapt_day_block(
        day_block: Dict[str, Any],
        *,
        force_replacement_first_exercise: bool,
        day_label_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        day_out = {
            "day": day_label_override if day_label_override is not None else day_block.get("day"),
            "focus": day_block.get("focus"),
            "exercises": [],
        }
        # Track exercises used within this day only (allow same exercise in different days if needed)
        day_used: Set[str] = set()

        for idx, templ_ex in enumerate(day_block.get("exercises", [])):
            templ_name = str(templ_ex.get("name"))
            templ_sets = (
                int(templ_ex.get("sets") or 0)
                if templ_ex.get("sets") is not None
                else None
            )
            templ_reps = (
                str(templ_ex.get("reps"))
                if templ_ex.get("reps") is not None
                else None
            )

            original_ex = find_exercise(templ_name)
            should_force = force_replacement_first_exercise and idx == 0

            keep_original = False
            if original_ex is not None and not should_force:
                ex_level = normalize_level(original_ex.get("level"))
                ex_equip = original_ex.get("equipment")
                ex_name_lower = (original_ex.get("name") or "").strip().lower()
                # Keep original only if compatible AND not already used in this day
                if (
                    level_distance(ex_level, user_level) <= 1
                    and equipment_compatible(ex_equip, user_equipment, user_group=user_group)
                    and ex_name_lower not in day_used
                ):
                    keep_original = True

            if keep_original:
                chosen = original_ex
                adaptation_note = None
            else:
                chosen = choose_replacement_exercise(
                    original_template_name=templ_name,
                    original_exercise=original_ex,
                    user_level=user_level,
                    user_equipment=user_equipment,
                    user_group=user_group,
                    already_used=day_used,  # Pass day-level used set
                )
                adaptation_note = {
                    "from_template_exercise": templ_name,
                    "to_exercise": chosen.get("name"),
                }

            chosen_name_lower = (chosen.get("name") or "").strip().lower()
            day_used.add(chosen_name_lower)
            plan_used_exercises.add(chosen_name_lower)

            chosen_level = normalize_level(chosen.get("level"))
            reps_out = templ_reps if templ_reps is not None else ""
            if templ_reps is not None and chosen_level is not None:
                reps_out = adapt_reps_string(
                    templ_reps, from_level=chosen_level, to_level=user_level
                )

            ex_out = {
                "name": chosen.get("name"),
                "category": chosen.get("category"),
                "level": chosen_level,
                "equipment": chosen.get("equipment"),
                "sets": templ_sets,
                "reps": reps_out,
                "source_template_exercise": None if keep_original else templ_name,
            }
            day_out["exercises"].append(ex_out)
            if adaptation_note:
                plan["adaptations"].append(adaptation_note)

        return day_out

    # Adapt base template days
    for day_block in template_schedule:
        plan["schedule"].append(
            adapt_day_block(day_block, force_replacement_first_exercise=False)
        )

    # Ensure exactly params.days_per_week days
    desired_days = int(params.days_per_week)
    if len(plan["schedule"]) > desired_days:
        plan["schedule"] = plan["schedule"][:desired_days]
    elif len(plan["schedule"]) < desired_days:
        if not template_schedule:
            raise RuntimeError("Template has no schedule blocks to adapt.")
        while len(plan["schedule"]) < desired_days:
            base_idx = len(plan["schedule"]) % len(template_schedule)
            base_day = template_schedule[base_idx]
            new_day = adapt_day_block(
                base_day,
                force_replacement_first_exercise=True,
                day_label_override=f"Day {len(plan['schedule']) + 1}",
            )
            plan["schedule"].append(new_day)

    return plan


def parse_constraints_from_params(params: ProgramBuilderParams) -> Dict[str, Any]:
    user_level = normalize_level(params.experience_level)
    user_equipment = normalize_equipment_input(params.equipment)
    user_group = classify_equipment_group(user_equipment)
    return {
        "goal": params.goal,
        "level": user_level,
        "equipment": sorted(user_equipment),
        "equipment_group": user_group,
        "days_per_week": params.days_per_week,
        "duration_weeks": params.duration_weeks,
    }


def build_weekly_plan(params: ProgramBuilderParams) -> Dict[str, Any]:
    templates = _load_workout_programs()
    constraints = parse_constraints_from_params(params)

    template = select_best_template(
        templates,
        goal=str(constraints["goal"]),
        level=str(constraints["level"]),
        equipment_group=str(constraints["equipment_group"]),
        days_per_week=int(constraints["days_per_week"]),
    )

    return adapt_weekly_plan_from_template(template, params=params)


build_custom_weekly_plan = build_weekly_plan
