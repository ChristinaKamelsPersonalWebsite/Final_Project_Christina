"""
Agent B — Nutrition Dietician Service
Framework : LlamaIndex
# LLM       : qwen2.5:1.5b via Ollama
RAG       : Qdrant 'nutrition_guides' collection
Port      : 8001
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent import NutritionAgent, get_nutrition_agent, warmup_nutrition_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_b")

app = FastAPI(
    title="Agent B — Nutrition Dietician",
    description="LlamaIndex-powered nutrition dietician. Uses Ollama + Qdrant RAG.",
    version="2.1.0",
)


class NutritionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from Agent A")
    user_query: str = Field(..., description="The user's nutrition question")
    user_context: Optional[dict] = Field(default=None)
    conversation_history: Optional[list[dict]] = Field(default=None)


class NutritionResponse(BaseModel):
    session_id: str
    answer: str
    sources_used: list[str] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)


#@app.on_event("startup")
#def startup_event():
 #   logger.info("Warming up NutritionAgent at startup...")
  #  warmup_nutrition_agent()
   # logger.info("NutritionAgent warmup complete.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "agent-b-nutrition-dietician",
        "framework": "llamaindex",
        "llm": "qwen2.5:1.5b@ollama",
            }


@app.post("/nutrition-advice", response_model=NutritionResponse)
def nutrition_advice(request: NutritionRequest):
    logger.info(
        "Nutrition request | session_id=%s | query=%.80s",
        request.session_id,
        request.user_query,
    )

    if not request.user_query.strip():
        raise HTTPException(status_code=422, detail="user_query must not be empty.")

    try:
        agent: NutritionAgent = get_nutrition_agent()
        result = agent.query(
            user_query=request.user_query,
            user_context=request.user_context or {},
            conversation_history=request.conversation_history or [],
        )
        return NutritionResponse(
            session_id=request.session_id,
            answer=result["answer"],
            sources_used=result.get("sources_used", []),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
        )
    except Exception as e:
        logger.error("Agent error | session_id=%s | %s", request.session_id, str(e))
        raise HTTPException(status_code=500, detail=f"Nutrition agent error: {str(e)}")


@app.post("/nutrition-advice/stream")
async def nutrition_advice_stream(request: NutritionRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=422, detail="user_query must not be empty.")

    async def generate():
        try:
            agent: NutritionAgent = get_nutrition_agent()
            result = agent.query(
                user_query=request.user_query,
                user_context=request.user_context or {},
                conversation_history=request.conversation_history or [],
            )
            answer = result.get("answer", "")
            import asyncio
            for word in answer.split(" "):
                yield word + " "
                await asyncio.sleep(0.01)
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )