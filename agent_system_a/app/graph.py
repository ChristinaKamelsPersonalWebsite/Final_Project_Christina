from __future__ import annotations

from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph

from agent_system_a.agents.exercise_agent import handle_exercise_query
from agent_system_a.tools.nutrition_client import make_nutrition_tool
from agent_system_a.agents.program_builder_agent import handle_program_query
from agent_system_a.app.config import get_settings
from agent_system_a.app.state import AgentState
from agent_system_a.app.supervisor import decide_next_agent

# Hard cap: supervisor cannot dispatch more than this many specialist calls per request.
MAX_ITERATIONS = 5


def supervisor_node(state: AgentState) -> AgentState:
    completed_agents = state.get("completed_agents", [])
    partial_results = state.get("partial_results", {})

    if len(completed_agents) >= MAX_ITERATIONS:
        next_agent = "finish"
    else:
        next_agent = decide_next_agent(
            message=state["message"],
            completed_agents=completed_agents,
            partial_results=partial_results,
        )
        if next_agent == "conversational":
            next_agent = "finish"
        elif next_agent != "finish" and next_agent in completed_agents:
            next_agent = "finish"

    return {
        **state,
        "completed_agents": completed_agents,
        "partial_results": partial_results,
        "next_agent": next_agent,
    }




def exercise_node(state: AgentState) -> AgentState:
    result = handle_exercise_query(state["message"])

    completed = list(state.get("completed_agents", []))
    if "exercise" not in completed:
        completed.append("exercise")

    partial = dict(state.get("partial_results", {}))
    partial["exercise"] = result

    return {
        **state,
        "completed_agents": completed,
        "partial_results": partial,
    }


def nutrition_node(state: AgentState) -> AgentState:
    nutrition_tool = make_nutrition_tool(
        session_id=state["session_id"],
        conversation_history=state.get("messages", []),
    )
    answer = nutrition_tool.invoke({
        "user_query": state["message"],
        "user_context": None,
    })
    result = {"response": answer, "route": "nutrition"}

    completed = list(state.get("completed_agents", []))
    if "nutrition" not in completed:
        completed.append("nutrition")

    partial = dict(state.get("partial_results", {}))
    partial["nutrition"] = result

    return {
        **state,
        "completed_agents": completed,
        "partial_results": partial,
    }


def program_node(state: AgentState) -> AgentState:
    result = handle_program_query(state["message"])

    completed = list(state.get("completed_agents", []))
    if "program" not in completed:
        completed.append("program")

    partial = dict(state.get("partial_results", {}))
    partial["program"] = result

    return {
        **state,
        "completed_agents": completed,
        "partial_results": partial,
    }


def progress_node(state: AgentState) -> AgentState:
    """
    Handles progress analysis requests.
    Parses workout logs from the user message and calls analyze_progress tool.
    """
    import re
    import json
    from agent_system_a.llm.ollama_client import chat_with_ollama

    message = state["message"]

    # Pre-process: split comma-separated sessions for better LLM parsing
    sessions = [s.strip() for s in message.split(',') if s.strip()]
    sessions_text = '\n'.join(f"- {s}" for s in sessions)

    extraction_prompt = f"""You are a workout log parser. Extract EVERY workout session listed below.

Sessions:
{sessions_text}

Rules:
- Extract ALL sessions.
- weight_kg: number before "kg". NEVER use 0 unless user says bodyweight.
- reps: number before "reps"
- sets: number before "sets"
- date: YYYY-MM-DD format exactly as written

Return ONLY a raw JSON array:
[{{"exercise": "Bench Press", "weight_kg": 80, "reps": 8, "sets": 3, "date": "2026-03-01"}}, ...]
"""

    try:
        raw = chat_with_ollama(
            system_prompt="You are a JSON extractor. Return only raw JSON array, no explanation.",
            user_prompt=extraction_prompt,
        )
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        logs_data = json.loads(clean)
    except Exception:
        logs_data = []

    # Call analyze_progress directly with parsed data
    from agent_system_a.tools.progress_tool import (
        analyze_progress, WorkoutLog, ProgressInput
    )

    if not logs_data:
        response_text = (
            "I couldn't find specific workout logs in your message. "
            "Please share your sessions like this:\n"
            "'Bench Press 80kg 8 reps 3 sets on 2026-03-01'\n"
            "Include exercise name, weight, reps, sets, and date for each session."
        )
    else:
        try:
            workout_logs = [WorkoutLog(**entry) for entry in logs_data]
            # Extract goal from message
            goal = None
            goal_keywords = {
                "strength": "strength", "muscle": "hypertrophy",
                "fat": "fat loss", "endurance": "endurance",
                "bench": "bench press", "squat": "squat", "deadlift": "deadlift"
            }
            for kw, goal_val in goal_keywords.items():
                if kw in message.lower():
                    goal = goal_val
                    break

            progress_input = ProgressInput(
                progress_logs=workout_logs,
                progress_goal=goal,
            )
            response_text = analyze_progress.invoke(progress_input.model_dump())
        except Exception as e:
            response_text = f"Error analyzing progress: {str(e)}"

    result = {"response": response_text, "route": "progress"}

    completed = list(state.get("completed_agents", []))
    if "progress" not in completed:
        completed.append("progress")

    partial = dict(state.get("partial_results", {}))
    partial["progress"] = result

    return {
        **state,
        "completed_agents": completed,
        "partial_results": partial,
    }

def finish_node(state: AgentState) -> AgentState:
    partial = state.get("partial_results", {})

    segments = []
    route_keys = []

    if "exercise" in partial:
        segments.append(partial["exercise"]["response"])
        route_keys.append("exercise")
    if "program" in partial:
        segments.append(partial["program"]["response"])
        route_keys.append("program")
    if "progress" in partial:
        segments.append(partial["progress"]["response"])
        route_keys.append("progress")
    if "nutrition" in partial:
        n_text = partial["nutrition"]["response"]
        if segments:
            segments.append(f"Nutrition guidance:\n{n_text}")
        else:
            segments.append(n_text)
        route_keys.append("nutrition")

    if segments:
        response = "\n\n".join(segments)
        route = "+".join(route_keys) if len(route_keys) > 1 else route_keys[0]
    else:
        # Check for conversational messages
        message = state.get("message", "").lower().strip()
        conversational = ["thanks", "thank you", "bye", "goodbye", "exit", "ok", "okay", "great", "cool", "gd bye"]
        if any(message == kw or message.startswith(kw) for kw in conversational):
            response = "You're welcome! Come back anytime you need fitness guidance. Stay consistent! 💪"
            route = "conversational"
        else:
            response = (
                "I'm not sure whether you want exercise help, nutrition guidance, "
                "or a workout program. Please be more specific."
            )
            route = "unknown"

    return {
        **state,
        "final_response": response,
        "final_route": route,
    }


def route_from_supervisor(state: AgentState) -> str:
    return state.get("next_agent", "finish")


def build_graph():
    settings = get_settings()

    try:
        checkpointer = RedisSaver.from_conn_string(settings.redis_url)
        checkpointer.setup()
    except Exception as e:
        print(f"[graph] Redis unavailable ({e}), falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("exercise", exercise_node)
    workflow.add_node("nutrition", nutrition_node)
    workflow.add_node("program", program_node)
    workflow.add_node("progress", progress_node)
    workflow.add_node("finish", finish_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "exercise": "exercise",
            "nutrition": "nutrition",
            "program": "program",
            "progress": "progress",
            "finish": "finish",
        },
    )

    workflow.add_edge("exercise", "supervisor")
    workflow.add_edge("nutrition", "supervisor")
    workflow.add_edge("program", "supervisor")
    workflow.add_edge("progress", "supervisor")
    workflow.add_edge("finish", END)

    return workflow.compile(checkpointer=checkpointer)