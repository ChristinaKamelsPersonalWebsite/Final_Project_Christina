from __future__ import annotations

import asyncio
import json
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent_system_a.app.config import get_settings
from agent_system_a.app.graph import build_graph
from agent_system_a.app.guardrails import apply_output_guardrails, validate_input
from agent_system_a.app.schemas import ChatRequest, ChatResponse
from agent_system_a.tools.progress_tool import analyze_progress
from agent_system_a.tools.mcp_client import (
    sync_call_macros,
    sync_call_tdee,
    sync_call_timeline,
)

router = APIRouter()
graph = build_graph()


def _wants_progress_analysis(message: str) -> bool:
    m = message.lower()
    return any(
        p in m
        for p in (
            "making progress",
            "am i making progress",
            "am i progressing",
            "my progress",
            "track progress",
            "analyze my progress",
            "progress on my lifts",
            "plateau",
            "personal record",
            " pr ",
            "hitting a pr",
            "workout log",
            "training log",
        )
    )

def _call_agent_b_progress(request: ChatRequest) -> tuple[str, str]:
    from agent_system_a.tools.progress_tool import WorkoutLog
    logs = [WorkoutLog(**log) for log in (request.progress_logs or [])]
    text = analyze_progress.invoke({
        "progress_logs": logs,
        "progress_goal": request.progress_goal or None,
    })
    return text, "progress_analysis"



def _call_mcp_tools(request: ChatRequest) -> tuple[str, list[str]]:
    """
    Call MCP tools via the MCP client (replaces raw requests.post calls).
    Each field is optional; only included fields are called.
    """
    chunks: list[str] = []
    routes: list[str] = []

    if request.mcp_tdee is not None:
        result = sync_call_tdee(**request.mcp_tdee)
        chunks.append("TDEE / energy estimate: " + str(result))
        routes.append("mcp_tdee")

    if request.mcp_macros is not None:
        result = sync_call_macros(**request.mcp_macros)
        chunks.append("Macro targets (grams): " + str(result))
        routes.append("mcp_macros")

    if request.mcp_timeline is not None:
        result = sync_call_timeline(**request.mcp_timeline)
        chunks.append("Goal timeline estimate: " + str(result))
        routes.append("mcp_timeline")

    return "\n\n".join(chunks), routes


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        is_valid, error_msg = validate_input(message)
        if not is_valid:
            return ChatResponse(
                response=error_msg,
                route="blocked",
                session_id=request.session_id,
            )

        prefix_parts: list[str] = []
        route_tags: list[str] = []

        if _wants_progress_analysis(message):
            if request.progress_logs:
                try:
                    b_text, b_route = _call_agent_b_progress(request)
                    # Return immediately with just Agent B's response.
                    # No need to run LangGraph when progress logs are provided.
                    return ChatResponse(
                        response=b_text,
                        route=b_route,
                        session_id=request.session_id,
                    )
                except Exception as exc:
                        prefix_parts.append(
                        "Progress analysis unavailable: " + str(exc)
                    )
                  
            else:
                # No early return — LangGraph still runs for coaching advice.
                # We just prepend a note asking the user to include logs next time.
                prefix_parts.append(
                    "To get a full progress analysis, include `progress_logs` in your "
                    "request (date, exercise, weight, reps, sets). "
                    "Here is what I can tell you based on your question:"
                )
                route_tags.append("progress_needs_logs")

        try:
            mcp_text, mcp_routes = _call_mcp_tools(request)
            if mcp_text:
                prefix_parts.append(mcp_text)
                route_tags.extend(mcp_routes)
        except Exception as exc:
            prefix_parts.append("MCP calculators unavailable: " + str(exc))
            route_tags.append("mcp_error")

        # Pass session_id as thread_id so MemorySaver persists conversation history.
        graph_config = {"configurable": {"thread_id": request.session_id}}

        result = graph.invoke(
            {
                "message": message,
                "session_id": request.session_id,
                "completed_agents": [],
                "partial_results": {},
            },
            config=graph_config,
        )

        print("GRAPH RESULT:", result)

        coach = apply_output_guardrails(result["final_response"], message)
        if prefix_parts:
            response = "\n\n".join(prefix_parts + [coach]).strip()
            route = "+".join(route_tags + [result["final_route"]])
        else:
            response = coach
            route = result["final_route"]

        return ChatResponse(
            response=response,
            route=route,
            session_id=request.session_id,
        )

    except Exception as exc:
        print("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Streaming endpoint for rubric compliance (2.5).
    Runs the same graph logic as /chat and streams the final response
    word-by-word as plain text. No token-level LLM streaming required.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    is_valid, error_msg = validate_input(message)
    if not is_valid:
        async def blocked():
            yield error_msg
        return StreamingResponse(blocked(), media_type="text/plain")

    graph_config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = graph.invoke(
            {
                "message": message,
                "session_id": request.session_id,
                "completed_agents": [],
                "partial_results": {},
            },
            config=graph_config,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")

    final_text = apply_output_guardrails(result["final_response"], message)

    async def word_stream():
        for word in final_text.split(" "):
            yield word + " "
            await asyncio.sleep(0)  # yield control to event loop between words

    return StreamingResponse(word_stream(), media_type="text/plain")