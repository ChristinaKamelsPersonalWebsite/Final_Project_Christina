from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    message: str
    session_id: Optional[str]

    messages: List[Dict[str, str]]  # conversation history

    completed_agents: List[str]
    partial_results: Dict[str, Any]

    next_agent: str
    final_response: str
    final_route: str
    error: str