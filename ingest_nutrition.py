"""
ingest_nutrition.py — Nutrition RAG Ingestion Script

Reads data/nutrition_guides.json, chunks by topic, embeds with
all-MiniLM-L6-v2, and upserts into Qdrant 'nutrition_guides' collection.

Chunking strategy: topic-based (one chunk per nutrition topic entry).
Each chunk is a self-contained nutritional concept, avoiding mid-topic
splits that would break context. Description, key_points, and examples
are concatenated into one rich text block per chunk.

Embedding model: all-MiniLM-L6-v2 (sentence-transformers, local)
Chosen for: speed, no API cost, 384-dim vectors, strong semantic
similarity performance on health/nutrition domain text.

Usage:
    python ingest_nutrition.py
    python ingest_nutrition.py --reset   # drops and recreates collection first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ingest_nutrition")

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL          = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME     = os.getenv("NUTRITION_COLLECTION", "nutrition_guides")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE         = 384  # all-MiniLM-L6-v2 output dimension
DATA_PATH           = Path(__file__).parent / "data" / "nutrition_guides.json"
BATCH_SIZE          = 32   # embed this many chunks at once


# ── Text builder ──────────────────────────────────────────────────────────────

def build_chunk_text(entry: dict) -> str:
    """
    Concatenates all meaningful text fields from one nutrition entry
    into a single rich text block for embedding.

    Structure:
        Topic: <topic>
        <description>
        Key points: <point1>. <point2>. ...
        Examples: <example1>. <example2>. ...
        Foods: <food1>, <food2>, ... (if present)
    """
    parts: list[str] = []

    topic = entry.get("topic", "").replace("_", " ").title()
    parts.append(f"Topic: {topic}")

    description = (entry.get("description") or "").strip()
    if description:
        parts.append(description)

    key_points = entry.get("key_points", [])
    if key_points:
        points_text = " ".join(f"{p.rstrip('.')}." for p in key_points if p)
        parts.append(f"Key points: {points_text}")

    examples = entry.get("examples", [])
    if examples:
        examples_text = " ".join(f"{e.rstrip('.')}." for e in examples if e)
        parts.append(f"Examples: {examples_text}")

    # Handle food_categories style entries with nested food lists
    foods = entry.get("foods", {})
    if foods:
        for category, food_list in foods.items():
            if food_list:
                label = category.replace("_", " ").title()
                parts.append(f"{label}: {', '.join(food_list)}.")

    return "\n".join(parts)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def wait_for_qdrant(retries: int = 10, delay: float = 3.0):
    """Wait until Qdrant is reachable — useful when running inside docker-compose."""
    import httpx
    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(f"{QDRANT_URL}/healthz", timeout=5.0)
            if r.status_code == 200:
                logger.info("Qdrant is ready.")
                return
        except Exception:
            pass
        logger.info("Waiting for Qdrant... attempt %d/%d", attempt, retries)
        time.sleep(delay)
    logger.error("Qdrant did not become ready in time. Exiting.")
    sys.exit(1)


def setup_collection(client, reset: bool = False):
    from qdrant_client.models import Distance, VectorParams

    collections = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION_NAME in collections:
        logger.info("Dropping existing collection '%s'...", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        collections = []

    if COLLECTION_NAME not in collections:
        logger.info("Creating collection '%s' (dim=%d)...", COLLECTION_NAME, VECTOR_SIZE)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        logger.info("Collection '%s' already exists — upserting into it.", COLLECTION_NAME)


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=False).tolist()


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(reset: bool = False):
    # 1. Wait for Qdrant
    wait_for_qdrant()
    client = get_qdrant_client()

    # 2. Setup collection
    setup_collection(client, reset=reset)

    # 3. Load data
    logger.info("Loading nutrition data from %s", DATA_PATH)
    if not DATA_PATH.exists():
        logger.error("Data file not found: %s", DATA_PATH)
        sys.exit(1)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    logger.info("Loaded %d nutrition entries.", len(entries))

    # 4. Load embedding model
    model = load_embedding_model()

    # 5. Build chunks
    chunks: list[dict] = []
    for i, entry in enumerate(entries):
        text = build_chunk_text(entry)
        if not text.strip():
            logger.warning("Entry %d has no text, skipping.", i)
            continue

        chunks.append({
            "id": i,
            "text": text,
            "topic": entry.get("topic", "unknown"),
            "source": "nutrition_guides.json",
            "source_type": "nutrition",        # used by Agent B metadata filter
        })

    logger.info("Built %d chunks for ingestion.", len(chunks))

    # 6. Embed and upsert in batches
    from qdrant_client.models import PointStruct

    total_upserted = 0
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = embed_batch(model, texts)

        points = [
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={
                    "text":        chunk["text"],
                    "topic":       chunk["topic"],
                    "source":      chunk["source"],
                    "source_type": chunk["source_type"],
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)
        logger.info(
            "Upserted batch %d-%d (%d points so far)",
            batch_start,
            batch_start + len(batch) - 1,
            total_upserted,
        )

    logger.info(
        "✅ Ingestion complete. %d chunks in collection '%s'.",
        total_upserted,
        COLLECTION_NAME,
    )

    # 7. Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(
        "Collection stats: %d vectors indexed.",
        info.vectors_count,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest nutrition data into Qdrant.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
