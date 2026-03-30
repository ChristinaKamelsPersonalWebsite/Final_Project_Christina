from __future__ import annotations


def calculate_tdee(
    age: int,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
) -> dict:
    sex = sex.lower().strip()
    activity_level = activity_level.lower().strip()

    if age <= 0 or weight_kg <= 0 or height_cm <= 0:
        raise ValueError("age, weight_kg, and height_cm must be positive")

    if sex not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")

    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    activity_map = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    multiplier = activity_map.get(activity_level, 1.55)
    tdee = bmr * multiplier

    return {
        "bmr": round(bmr, 2),
        "tdee": round(tdee, 2),
        "activity_multiplier": multiplier,
    }


def calculate_macros(calories: int, goal: str) -> dict:
    goal = goal.lower().strip()

    if calories <= 0:
        raise ValueError("calories must be positive")

    if goal in {"fat loss", "cutting", "weight loss"}:
        protein_pct, carbs_pct, fats_pct = 0.35, 0.35, 0.30
    elif goal in {"strength", "muscle gain", "hypertrophy", "bulking"}:
        protein_pct, carbs_pct, fats_pct = 0.30, 0.45, 0.25
    else:
        protein_pct, carbs_pct, fats_pct = 0.30, 0.40, 0.30

    protein_g = (calories * protein_pct) / 4
    carbs_g = (calories * carbs_pct) / 4
    fats_g = (calories * fats_pct) / 9

    return {
        "protein_g": round(protein_g, 2),
        "carbs_g": round(carbs_g, 2),
        "fats_g": round(fats_g, 2),
    }


def estimate_goal_timeline(
    current_weight_kg: float,
    target_weight_kg: float,
    weekly_change_kg: float,
) -> dict:
    if current_weight_kg <= 0 or target_weight_kg <= 0:
        raise ValueError("current_weight_kg and target_weight_kg must be positive")

    if weekly_change_kg <= 0:
        raise ValueError("weekly_change_kg must be positive")

    diff = abs(current_weight_kg - target_weight_kg)
    weeks = diff / weekly_change_kg

    return {
        "weight_difference_kg": round(diff, 2),
        "estimated_weeks": round(weeks, 1),
    }