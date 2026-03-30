from __future__ import annotations

from pydantic import BaseModel, Field


class TDEERequest(BaseModel):
    age: int = Field(..., ge=10, le=100)
    sex: str
    weight_kg: float = Field(..., ge=20)
    height_cm: float = Field(..., ge=100)
    activity_level: str


class MacrosRequest(BaseModel):
    calories: int = Field(..., ge=800)
    goal: str


class TimelineRequest(BaseModel):
    current_weight_kg: float = Field(..., ge=20)
    target_weight_kg: float = Field(..., ge=20)
    weekly_change_kg: float = Field(..., gt=0)