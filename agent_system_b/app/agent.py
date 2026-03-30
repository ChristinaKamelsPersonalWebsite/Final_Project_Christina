"""
NutritionAgent — LlamaIndex-based nutrition dietician.

Flow:
  1. Retrieve relevant chunks from Qdrant (nutrition_guides collection)
  2. Build a prompt with retrieved context + user question
  3. Call qwen2.5:1.5b via Ollama to synthesize a coach-style answer
"""
import os
import logging
import threading
from typing import Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger("nutrition_agent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("NUTRITION_COLLECTION", "nutrition_guides")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_SYSTEM_PROMPT = """You are a certified sports dietician and nutrition coach.
Answer the user's nutrition question based ONLY on the retrieved context below.
Be practical, specific, and motivating. Give concrete numbers where relevant.
Do not invent information not supported by the context.
If the user has shared their profile (weight, goal, restrictions), personalize your answer.
End your answer with 2-3 follow-up questions the user might find useful, prefixed with 'Follow-up questions:'"""

_index_lock = threading.Lock()
_shared_agent: Optional["NutritionAgent"] = None


def _build_index() -> VectorStoreIndex:
    """Build the LlamaIndex vector store index once."""
    logger.info("Initializing LlamaIndex with Ollama + Qdrant...")

    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=0.3,
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    logger.info("LlamaIndex index ready.")
    return index


class NutritionAgent:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="source_type",
                        value="nutrition",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
        )

    def query(
        self,
        user_query: str,
        user_context: dict,
        conversation_history: list[dict],
    ) -> dict:
        try:
            context_lines = []
            labels = {
                "age": "Age",
                "weight_kg": "Weight",
                "height_cm": "Height",
                "goal": "Goal",
                "dietary_restrictions": "Dietary restrictions",
                "tdee": "TDEE",
                "macros": "Macro targets",
                "activity_level": "Activity level",
            }

            for key, label in labels.items():
                if user_context.get(key):
                    context_lines.append(f"{label}: {user_context[key]}")

            profile_block = ""
            if context_lines:
                profile_block = "User profile:\n" + "\n".join(context_lines) + "\n\n"

            history_block = ""
            if conversation_history:
                lines = []
                for msg in conversation_history[-4:]:
                    role = msg.get("role", "user").capitalize()
                    lines.append(f"{role}: {msg.get('content', '')}")
                history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

            full_query = (
                f"{_SYSTEM_PROMPT}\n\n"
                f"{profile_block}"
                f"{history_block}"
                f"User question: {user_query}"
            )

            response = self.query_engine.query(full_query)
            answer_text = str(response)

            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    topic = node.metadata.get("topic", "")
                    source = node.metadata.get("source", "nutrition guide")
                    if topic:
                        sources.append(f"{source} ({topic})")
                    else:
                        sources.append(source)

            follow_ups = []
            if "Follow-up questions:" in answer_text:
                parts = answer_text.split("Follow-up questions:")
                answer_clean = parts[0].strip()
                fq_text = parts[1].strip() if len(parts) > 1 else ""
                for line in fq_text.split("\n"):
                    line = line.strip().lstrip("-•123456789. ")
                    if line:
                        follow_ups.append(line)
            else:
                answer_clean = answer_text

            return {
                "answer": answer_clean,
                "sources_used": list(dict.fromkeys(sources))[:5],
                "follow_up_suggestions": follow_ups[:3],
            }

        except Exception as e:
            logger.error("NutritionAgent query failed: %s", str(e))
            return {
                "answer": (
                    "I'm having trouble accessing the nutrition knowledge base right now. "
                    "Please try again in a moment."
                ),
                "sources_used": [],
                "follow_up_suggestions": [],
            }


def get_nutrition_agent() -> NutritionAgent:
    global _shared_agent

    if _shared_agent is None:
        with _index_lock:
            if _shared_agent is None:
                index = _build_index()
                _shared_agent = NutritionAgent(index=index)

    return _shared_agent


def warmup_nutrition_agent() -> None:
    """Force model/index initialization at app startup."""
    get_nutrition_agent()