from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8002/mcp")


@asynccontextmanager
async def _mcp_session():
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_calculate_tdee(age, sex, weight_kg, height_cm, activity_level):
    async with _mcp_session() as session:
        result = await session.call_tool(
            "calculate_tdee",
            arguments={"age": age, "sex": sex, "weight_kg": weight_kg,
                       "height_cm": height_cm, "activity_level": activity_level},
        )
        return json.loads(result.content[0].text)


async def call_calculate_macros(calories, goal):
    async with _mcp_session() as session:
        result = await session.call_tool(
            "calculate_macros",
            arguments={"calories": calories, "goal": goal},
        )
        return json.loads(result.content[0].text)


async def call_estimate_goal_timeline(current_weight_kg, target_weight_kg, weekly_change_kg):
    async with _mcp_session() as session:
        result = await session.call_tool(
            "estimate_goal_timeline",
            arguments={"current_weight_kg": current_weight_kg,
                       "target_weight_kg": target_weight_kg,
                       "weekly_change_kg": weekly_change_kg},
        )
        return json.loads(result.content[0].text)


def sync_call_tdee(**kwargs) -> dict:
    return asyncio.run(call_calculate_tdee(**kwargs))


def sync_call_macros(**kwargs) -> dict:
    return asyncio.run(call_calculate_macros(**kwargs))


def sync_call_timeline(**kwargs) -> dict:
    return asyncio.run(call_estimate_goal_timeline(**kwargs))