"""
MCP Server — Body Metrics & Goal Service
Wraps existing calculator logic as proper MCP tools.
Transport: streamable-http (HTTP + SSE), runs on port 8002.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.app.calculators import (
    calculate_macros,
    calculate_tdee,
    estimate_goal_timeline,
)

mcp = FastMCP("body-metrics-goal-service", port=8002, host="0.0.0.0")

@mcp.tool()
def calculate_tdee_tool(
    age: int,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
) -> dict:
    return calculate_tdee(
        age=age,
        sex=sex,
        weight_kg=weight_kg,
        height_cm=height_cm,
        activity_level=activity_level,
    )


@mcp.tool()
def calculate_macros_tool(calories: int, goal: str) -> dict:
    return calculate_macros(calories=calories, goal=goal)


@mcp.tool()
def estimate_goal_timeline_tool(
    current_weight_kg: float,
    target_weight_kg: float,
    weekly_change_kg: float,
) -> dict:
    return estimate_goal_timeline(
        current_weight_kg=current_weight_kg,
        target_weight_kg=target_weight_kg,
        weekly_change_kg=weekly_change_kg,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")