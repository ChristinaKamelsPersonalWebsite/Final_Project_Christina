from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field




class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to coach the athlete.")
    session_id: Optional[str] = Field(
        None, description="Optional existing session identifier."
    )
    progress_logs: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional workout logs for Agent B progress analysis (exercise_name, date, sets, reps, weight).",
    )
    progress_goal: Optional[str] = Field(
        None, description="Optional training goal passed through to Agent B."
    )
    mcp_tdee: Optional[Dict[str, Any]] = Field(
        None, description="Optional body for MCP POST /calculate-tdee (age, sex, weight_kg, height_cm, activity_level)."
    )
    mcp_macros: Optional[Dict[str, Any]] = Field(
        None, description="Optional body for MCP POST /calculate-macros (calories, goal)."
    )
    mcp_timeline: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional body for MCP POST /estimate-goal-timeline (current_weight_kg, target_weight_kg, weekly_change_kg).",
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Coaching response text.")
    route: str = Field(..., description="Internal routing/agent identifier.")
    session_id: Optional[str] = Field(
        None, description="Session identifier used for this conversation."
    )


class ExerciseSearchParams(BaseModel):
    query: str = Field(..., description="Free-text description of the exercise need.")
    
    muscle_group: Optional[str] = Field(
        None, description="Target muscle group (e.g., chest, back, legs)."
    )
    
    equipment: Optional[str] = Field(
        None, description="Preferred equipment (e.g., dumbbells, barbell, machines)."
    )

    difficulty: Optional[str] = Field(
        None, description="Difficulty level such as beginner, intermediate, or expert."
    )

    category: Optional[str] = Field(
        None, description="Exercise category such as strength or strongman."
    )

    limit: int = Field(
        10, ge=1, le=100, description="Max number of exercises to return."
    )


class ProgramBuilderParams(BaseModel):
    goal: str = Field(..., description="Fitness goal (e.g., strength, hypertrophy, fat loss).")
    days_per_week: int = Field(3, ge=1, le=7, description="Training frequency per week.")
    duration_weeks: int = Field(4, ge=1, le=52, description="Program length in weeks.")
    experience_level: Optional[str] = Field(
        None, description="Experience level (e.g., beginner, intermediate, advanced)."
    )
    equipment: Optional[list[str]] = Field(
        None, description="List of available equipment names."
    )


class NutritionSearchParams(BaseModel):
    goal: str = Field(..., description="Nutrition goal (e.g., maintenance, cutting, bulking).")
    dietary_preference: Optional[str] = Field(
        None, description="Dietary preference (e.g., vegetarian, halal, omnivore)."
    )
    calories_target: Optional[int] = Field(
        None, ge=0, description="Optional calorie target for meal suggestions."
    )
    limit: int = Field(10, ge=1, le=100, description="Max number of nutrition suggestions to return.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message suitable for API clients.")
    route: Optional[str] = Field(None, description="Optional internal route/agent identifier.")
    session_id: Optional[str] = Field(
        None, description="Optional session identifier associated with the error."
    )

