"""
analyze_progress — LangGraph tool for Agent A

Replaces the old Agent B HTTP call for progress analysis.
All logic is deterministic (same math as before), but the output is a
smooth, readable coaching narrative instead of a raw structured dict.

Register in your tool list:
    from agent_system_a.tools.progress_tool import analyze_progress
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("progress_tool")


# ── Input schema ──────────────────────────────────────────────────────────────

class WorkoutLog(BaseModel):
    exercise:  str   = Field(..., description="Exercise name, e.g. 'Bench Press'")
    weight_kg: float = Field(..., description="Weight used in kg (use 0 for bodyweight)")
    reps:      int   = Field(..., description="Reps completed per set")
    sets:      int   = Field(..., description="Number of sets completed")
    date:      str   = Field(..., description="Date in YYYY-MM-DD format")


class ProgressInput(BaseModel):
    progress_logs: list[WorkoutLog] = Field(
        ..., description="All workout log entries to analyze"
    )
    progress_goal: Optional[str] = Field(
        default=None,
        description="User's training goal, e.g. 'increase squat', 'lose weight', 'build muscle'",
    )


# ── Math helpers ──────────────────────────────────────────────────────────────

def _epley_1rm(weight: float, reps: int) -> float:
    """Epley formula: estimated 1-rep max."""
    if reps <= 1:
        return weight
    return round(weight * (1 + reps / 30), 1)


def _group_by_exercise(logs: list[WorkoutLog]) -> dict[str, list[WorkoutLog]]:
    grouped: dict[str, list[WorkoutLog]] = defaultdict(list)
    for log in sorted(logs, key=lambda x: x.date):
        grouped[log.exercise].append(log)
    return dict(grouped)


def _progression(logs: list[WorkoutLog]) -> dict:
    first, last = logs[0], logs[-1]
    first_1rm = _epley_1rm(first.weight_kg, first.reps)
    last_1rm  = _epley_1rm(last.weight_kg,  last.reps)
    pct = ((last_1rm - first_1rm) / first_1rm * 100) if first_1rm else 0.0
    return {
        "first_date": first.date,
        "last_date":  last.date,
        "first_1rm":  first_1rm,
        "last_1rm":   last_1rm,
        "change_pct": round(pct, 1),
        "sessions":   len(logs),
    }


def _best_pr(logs: list[WorkoutLog]) -> tuple[float, str]:
    best = max(logs, key=lambda l: _epley_1rm(l.weight_kg, l.reps))
    return _epley_1rm(best.weight_kg, best.reps), best.date


def _is_plateau(logs: list[WorkoutLog], window: int = 3) -> bool:
    """
    True if the last `window` sessions show <2.5% change in estimated 1RM.
    Requires at least window+1 sessions to make a judgment.
    """
    if len(logs) < window + 1:
        return False
    recent    = logs[-(window + 1):]
    first_1rm = _epley_1rm(recent[0].weight_kg, recent[0].reps)
    last_1rm  = _epley_1rm(recent[-1].weight_kg, recent[-1].reps)
    if first_1rm == 0:
        return False
    return abs((last_1rm - first_1rm) / first_1rm * 100) < 2.5


def _weekly_volume(logs: list[WorkoutLog]) -> list[tuple[str, float]]:
    weekly: dict[str, float] = defaultdict(float)
    for log in logs:
        week = datetime.strptime(log.date, "%Y-%m-%d").strftime("%Y-W%V")
        weekly[week] += log.weight_kg * log.reps * log.sets
    return sorted(weekly.items())


# ── Narrative renderer ────────────────────────────────────────────────────────

def _build_narrative(
    grouped: dict[str, list[WorkoutLog]],
    goal: Optional[str],
) -> str:
    lines: list[str] = []
    plateaued: list[str] = []
    progressing: list[str] = []

    goal_note = f" toward your goal of **{goal}**" if goal else ""
    lines.append(f"Here's a breakdown of your training progress{goal_note}:\n")

    for exercise, logs in grouped.items():
        prog = _progression(logs)
        pr_1rm, pr_date = _best_pr(logs)
        on_plateau = _is_plateau(logs)

        if prog["change_pct"] > 2.5:
            icon, trend_word = "📈", "improved"
            progressing.append(exercise)
        elif prog["change_pct"] < -2.5:
            icon, trend_word = "📉", "declined"
        else:
            icon, trend_word = "➡️", "stayed roughly flat"

        lines.append(f"**{exercise}** {icon}")
        lines.append(
            f"  Over {prog['sessions']} session(s) "
            f"({prog['first_date']} → {prog['last_date']}), "
            f"your estimated 1RM has {trend_word}: "
            f"{prog['first_1rm']} kg → {prog['last_1rm']} kg "
            f"({'+' if prog['change_pct'] >= 0 else ''}{prog['change_pct']}%)."
        )
        lines.append(
            f"  Your personal record is **{pr_1rm} kg** (est. 1RM), set on {pr_date}."
        )

        if on_plateau:
            plateaued.append(exercise)
            lines.append(
                "  ⚠️  **Plateau detected** — your last few sessions haven't shown "
                "meaningful improvement. Consider a deload week, adjusting your rep ranges, "
                "or adding a variation of this movement."
            )

        lines.append("")

    # Volume trend across all exercises
    all_logs = [log for logs in grouped.values() for log in logs]
    weekly = _weekly_volume(all_logs)
    if len(weekly) >= 2:
        w0, v0 = weekly[0]
        wn, vn = weekly[-1]
        direction = "increased" if vn > v0 else "decreased"
        lines.append(
            f"**Overall training volume** has {direction} from "
            f"{v0:,.0f} kg-reps (week {w0}) to {vn:,.0f} kg-reps (week {wn}).\n"
        )

    # Summary
    lines.append("**Summary**")
    if progressing:
        lines.append(f"  ✅ Solid progress on: {', '.join(progressing)}. Keep pushing!")
    if plateaued:
        lines.append(
            f"  ⚠️  Plateau detected on: {', '.join(plateaued)}. Time to change the stimulus."
        )
    if not progressing and not plateaued:
        lines.append(
            "  Not enough data yet for strong conclusions. "
            "Log at least 4 sessions per exercise for a reliable analysis."
        )

    # Goal-specific coaching tip
    if goal:
        g = goal.lower()
        if any(k in g for k in ["strength", "bench", "squat", "deadlift", "press", "pull"]):
            lines.append(
                "\n💡 **Strength tip:** Aim for +2.5 kg or +1 rep per week. "
                "If you've stalled, try wave loading or a 5/3/1 progression scheme."
            )
        elif any(k in g for k in ["muscle", "hypertrophy", "size", "bulk"]):
            lines.append(
                "\n💡 **Hypertrophy tip:** Total weekly volume drives muscle growth. "
                "If gains have stalled, add a set per exercise before increasing weight."
            )
        elif any(k in g for k in ["lose", "weight", "fat", "cut", "deficit"]):
            lines.append(
                "\n💡 **Cut tip:** In a calorie deficit, maintaining your current weights "
                "is already a win — you're preserving muscle. Don't expect PRs while cutting."
            )
        elif any(k in g for k in ["endurance", "marathon", "cardio", "run"]):
            lines.append(
                "\n💡 **Endurance tip:** For endurance goals, track weekly volume and "
                "session RPE — consistency matters more than 1RM increases."
            )

    return "\n".join(lines)


# ── LangGraph tool ────────────────────────────────────────────────────────────

@tool("analyze_progress", args_schema=ProgressInput)
def analyze_progress(
    progress_logs: list[WorkoutLog],
    progress_goal: Optional[str] = None,
) -> str:
    """
    Analyzes the user's workout history and returns a clear progress report.

    Use when the user asks:
      - "Am I making progress?"
      - "Have I plateaued on bench press?"
      - "How is my training going?"
      - "Can you check my workout logs?"
      - "Am I getting stronger?"

    Input:
      progress_logs  — list of workout entries (exercise, weight_kg, reps, sets, date)
      progress_goal  — optional training goal string for personalized tips

    Output:
      A readable, coach-style progress narrative with trends, PRs,
      plateau detection, volume trends, and goal-specific coaching tips.
    """
    logger.info("Progress analysis | logs=%d | goal=%s", len(progress_logs), progress_goal)

    if not progress_logs:
        return (
            "I don't have any workout logs to analyze yet. "
            "Share your recent sessions — exercise name, weight, reps, sets, "
            "and date — and I'll give you a full breakdown."
        )

    if len(progress_logs) < 3:
        return (
            f"You've logged {len(progress_logs)} session(s) so far — good start! "
            "For a meaningful analysis I need at least 3-4 sessions. "
            "Keep logging and check back soon."
        )

    try:
        grouped   = _group_by_exercise(progress_logs)
        narrative = _build_narrative(grouped, progress_goal)
        return narrative
    except Exception as e:
        logger.error("Progress analysis error: %s", str(e))
        return (
            "Something went wrong while analyzing your logs. "
            "Make sure all entries have a valid exercise name, weight (0 for bodyweight), "
            "reps, sets, and a date in YYYY-MM-DD format, then try again."
        )
