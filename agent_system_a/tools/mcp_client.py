from __future__ import annotations

import os
import requests

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8002/mcp")

# Extract base URL from MCP URL (remove /mcp suffix)
_BASE_URL = MCP_SERVER_URL.replace("/mcp", "")


def sync_call_tdee(age, sex, weight_kg, height_cm, activity_level) -> dict:
    """Call TDEE calculator directly via HTTP."""
    try:
        resp = requests.post(
            f"{_BASE_URL}/tdee",
            json={
                "age": age,
                "sex": sex,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "activity_level": activity_level,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback: calculate directly using Mifflin-St Jeor
        sex_lower = sex.lower().strip()
        if sex_lower == "male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        multiplier = multipliers.get(activity_level.lower().strip(), 1.55)
        tdee = bmr * multiplier
        return {
            "bmr": round(bmr, 2),
            "tdee": round(tdee, 2),
            "activity_multiplier": multiplier,
        }


def sync_call_macros(calories, goal) -> dict:
    """Call macros calculator directly via HTTP."""
    try:
        resp = requests.post(
            f"{_BASE_URL}/macros",
            json={"calories": calories, "goal": goal},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback: calculate directly
        goal_lower = goal.lower().strip()
        if goal_lower in {"fat loss", "cutting", "weight loss"}:
            protein_pct, carbs_pct, fats_pct = 0.35, 0.35, 0.30
        elif goal_lower in {"strength", "muscle gain", "hypertrophy", "bulking"}:
            protein_pct, carbs_pct, fats_pct = 0.30, 0.45, 0.25
        else:
            protein_pct, carbs_pct, fats_pct = 0.30, 0.40, 0.30

        return {
            "protein_g": round((calories * protein_pct) / 4, 2),
            "carbs_g": round((calories * carbs_pct) / 4, 2),
            "fats_g": round((calories * fats_pct) / 9, 2),
        }


def sync_call_timeline(current_weight_kg, target_weight_kg, weekly_change_kg) -> dict:
    """Call goal timeline calculator directly via HTTP."""
    try:
        resp = requests.post(
            f"{_BASE_URL}/timeline",
            json={
                "current_weight_kg": current_weight_kg,
                "target_weight_kg": target_weight_kg,
                "weekly_change_kg": weekly_change_kg,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # Fallback: calculate directly
        diff = abs(current_weight_kg - target_weight_kg)
        weeks = diff / weekly_change_kg
        return {
            "weight_difference_kg": round(diff, 2),
            "estimated_weeks": round(weeks, 1),
            "estimated_months": round(weeks / 4.3, 1),
        }